// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>

#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include <immintrin.h>

#include "submatrices.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace std;

#define TILE_WIDTH 32 // bfloats
#define TILE_HEIGHT 32 // bfloats
#define TILE_SIZE 2048 // bfloats

// I wanto to have all the SRAM available, maybe I could use also multiple CBs
#define SRAM_TILES 64

int axpy_ttker(
    vector<bfloat16>& up,
    vector<bfloat16>& left,
    vector<bfloat16>& right,
    vector<bfloat16>& down,
    vector<bfloat16>& scalar,
    vector<bfloat16>& output,
    uint32_t n_tiles,
    uint32_t rows,
    uint32_t cols,
    uint32_t iterations,
    distributed::MeshDevice* mesh_device  // [CHANGED] was: IDevice* device
) {

    //* ---------------------------------------------------------
    //* HOST INITIALIZATION
    //* ---------------------------------------------------------

    // Ensure printing from kernel is enabled (so we can see the output of the Data Movement kernels).
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of the Data "
            "Movement kernels.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0\n");
    }

    // [CHANGED] The MeshDevice is now created once in main() and passed in.
    // Previously main() opened device 0 with CreateDevice() and this function opened it a second time
    // with create_unit_mesh(0), which fails at runtime.

    // In Metalium, submitting operations to the device is done through a command queue. This includes
    // uploading/downloading data to/from the device, and executing programs.
    // A MeshCommandQueue is a software concept that allows developers to submit operations to a MeshDevice.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // A MeshWorkload is a collection of programs that are executed on a MeshDevice.
    // The specific physical devices that the workload is executed on are determined by the MeshCoordinateRange.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    // A Program contains kernels that perform computations or data movement.


    Program program = CreateProgram();

    DataFormat data_format = DataFormat::Float16_b;
    // [CHANGED] qualified: the global-namespace MathFidelity alias is deprecated
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    const auto core_grid = mesh_device->compute_with_storage_grid_size();  // [CHANGED] was: device->

    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] = split_work_to_cores(core_grid, n_tiles);

    cout << " Number of tensixes: " << num_cores << endl;
    cout << " Tensixes group 1 has " << work_per_core1 << " tiles" << endl;
    cout << " Tensixes group 2 has " << work_per_core2 << " tiles" << endl;

    //? If you have 64 cores, you at least 64 tiles so at least 32x64: if you have less you need to reduce cores
    //? I need to understand if it is better to have 1 tile per 64 core or less cores more tiles per core [MEASURE]

    // ---------------------------------------------------------
    // DRAM BUFFER CREATION: OFFCHIP GDDR6 MEMORY 12GB
    // ---------------------------------------------------------

    cout << "Creating DRAM buffers..." << endl;

    // [CHANGED] InterleavedBufferConfig + CreateBuffer (deprecated) -> MeshBuffer.
    // The old single config is split in two:
    //  - DeviceLocalBufferConfig: page_size and buffer_type (it has NO .size field)
    //  - ReplicatedBufferConfig:  size in bytes, per device
    // The device becomes the third argument of MeshBuffer::create.
    // DRAM buffers created this way are interleaved, as before.
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = TILE_SIZE,
        .buffer_type = BufferType::DRAM};

    distributed::ReplicatedBufferConfig dram_inout_config{
        .size = up.size() * sizeof(bfloat16)};

    distributed::ReplicatedBufferConfig dram_scalar_config{
        .size = scalar.size() * sizeof(bfloat16)};

    std::shared_ptr<distributed::MeshBuffer> input_dram_buffer = distributed::MeshBuffer::create(dram_inout_config, dram_config, mesh_device);
    std::shared_ptr<distributed::MeshBuffer> up_dram_buffer = distributed::MeshBuffer::create(dram_inout_config, dram_config, mesh_device);
    std::shared_ptr<distributed::MeshBuffer> left_dram_buffer = distributed::MeshBuffer::create(dram_inout_config, dram_config, mesh_device);
    std::shared_ptr<distributed::MeshBuffer> right_dram_buffer = distributed::MeshBuffer::create(dram_inout_config, dram_config, mesh_device);
    std::shared_ptr<distributed::MeshBuffer> down_dram_buffer = distributed::MeshBuffer::create(dram_inout_config, dram_config, mesh_device);
    std::shared_ptr<distributed::MeshBuffer> output_dram_buffer = distributed::MeshBuffer::create(dram_inout_config, dram_config, mesh_device);
    std::shared_ptr<distributed::MeshBuffer> scalar_dram_buffer = distributed::MeshBuffer::create(dram_scalar_config, dram_config, mesh_device);

    // ! In this case you should declare the starting address and the size of the region in the shared cache
    // ! Then Is necessary a method to avoid contention, but for stencil could be avoided by the nature of the problem

    // ---------------------------------------------------------
    // SRAM BUFFER CREATION: TOTAL SHARED BETWEEN CORES 108MB HIGH-SPEED REGISTERS
    // ---------------------------------------------------------

    cout << "Creating SRAM buffers..." << endl;

    constexpr uint32_t cb_indices[] = {
        CBIndex::c_0, // input
        CBIndex::c_1, // up
        CBIndex::c_2, // left
        CBIndex::c_3, // right
        CBIndex::c_4, // down
        CBIndex::c_5, // scalar
        CBIndex::c_6, // auxiliary
        CBIndex::c_7, // output
    };

    // Circular Buffers (CB) have to be created rising order!
    for(int i = 0; i < 8; i++) {

        CircularBufferConfig cb_config(TILE_SIZE * SRAM_TILES,
                                            {{cb_indices[i], data_format}}
        );
        cb_config.set_page_size(cb_indices[i], TILE_SIZE);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    }

    // ---------------------------------------------------------
    // KERNELS CREATION: We need a reader, writer and then a compute
    // ---------------------------------------------------------

    cout << "Creating kernels..." << endl;


    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*scalar_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*up_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*left_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*right_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*down_dram_buffer).append_to(reader_compile_time_args);

    auto reader_kernel_id = tt_metal::CreateKernel( 
        program, 
        OVERRIDE_KERNEL_PREFIX "axpy_stencil/kernels/dataflow/reader_input.cpp",
        all_cores, 
        tt_metal::DataMovementConfig{ 
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args
        }
    );

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*output_dram_buffer).append_to(writer_compile_time_args);

    auto writer_kernel_id = tt_metal::CreateKernel(
        program, 
        OVERRIDE_KERNEL_PREFIX "axpy_stencil/kernels/dataflow/writer_output.cpp",
        all_cores, 
        tt_metal::DataMovementConfig{ 
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args
        }
    );


    std::vector<uint32_t> compute_args = {};
    KernelHandle stencil_kernel_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "axpy_stencil/kernels/compute/stencil.cpp",
        all_cores,
        tt_metal::ComputeConfig {
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_args,
        }
    );

    // Compute config has a lot of arguments, but I don't know them now:
    // - .math_approx_modes
    // - .fp32_dest_acc_en
    // - .defines
    // If think many others

    // ---------------------------------------------------------
    // SETUP RUNTIME ARGS: Set the runtime arguments for the kernels
    // ---------------------------------------------------------

    cout << "Setting up runtime arguments..." << endl;

    // [CHANGED] MeshBuffer::address() returns a 64-bit DeviceAddr, but runtime args are uint32_t.
    // Putting address() directly inside SetRuntimeArgs(..., { ... }) is a narrowing error under Clang
    // (tt-metal's own examples only build because their CMake adds -Wno-c++11-narrowing).
    const uint32_t up_addr     = static_cast<uint32_t>(up_dram_buffer->address());
    const uint32_t left_addr   = static_cast<uint32_t>(left_dram_buffer->address());
    const uint32_t right_addr  = static_cast<uint32_t>(right_dram_buffer->address());
    const uint32_t down_addr   = static_cast<uint32_t>(down_dram_buffer->address());
    const uint32_t scalar_addr = static_cast<uint32_t>(scalar_dram_buffer->address());
    const uint32_t output_addr = static_cast<uint32_t>(output_dram_buffer->address());

    uint32_t start_tile_idx = 0;  // [CHANGED] was int (same narrowing issue)
    for(const auto& core_range : core_group_1.ranges()){
        for(const auto& core : core_range) {
            const uint32_t core_id = static_cast<uint32_t>(core.x*8 + core.y);  // [CHANGED] CoreCoord x/y are size_t
            tt_metal::SetRuntimeArgs( program, reader_kernel_id, core, {
                up_addr,
                left_addr,
                right_addr,
                down_addr,
                scalar_addr,
                start_tile_idx,
                core_id,
                work_per_core1
            });

            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, {
                output_addr, start_tile_idx, work_per_core1
            });

            tt_metal::SetRuntimeArgs(program, stencil_kernel_id, core, {
                work_per_core1
            });

            start_tile_idx += work_per_core1;
        }
    }
    for(const auto& core_range : core_group_2.ranges()){
        for(const auto& core : core_range) {
            const uint32_t core_id = static_cast<uint32_t>(core.x*8 + core.y);
            tt_metal::SetRuntimeArgs( program, reader_kernel_id, core, {
                up_addr,
                left_addr,
                right_addr,
                down_addr,
                scalar_addr,
                start_tile_idx,
                core_id,
                work_per_core2
            });

            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, {
                output_addr, start_tile_idx, work_per_core2
            });

            tt_metal::SetRuntimeArgs(program, stencil_kernel_id, core, {
                work_per_core2
            });

            start_tile_idx += work_per_core2;
        }
    }

    // [CHANGED] Add the program to the workload ONCE, outside the iteration loop.
    // Calling add_program() again for the same device range throws
    // ("Program range ... overlaps with the previously added range"), and after the first
    // std::move the program object is empty anyway. The same workload can be enqueued many times.
    workload.add_program(device_range, std::move(program));


    // ---------------------------------------------------------
    // ENQUEUE WRITE BUFFERS: Write data on the allocated buffers
    // ---------------------------------------------------------

    cout << "Memcpy and compute launch..." << endl;

    // [CHANGED] distributed::EnqueueWriteMeshBuffer takes a std::vector, not a raw pointer.
    // For raw pointers (needed in the loop below) use cq.enqueue_write_mesh_buffer(...).
    distributed::EnqueueWriteMeshBuffer(cq, up_dram_buffer, up, true);
    distributed::EnqueueWriteMeshBuffer(cq, left_dram_buffer, left, true);
    distributed::EnqueueWriteMeshBuffer(cq, right_dram_buffer, right, true);
    distributed::EnqueueWriteMeshBuffer(cq, down_dram_buffer, down, true);
    distributed::EnqueueWriteMeshBuffer(cq, scalar_dram_buffer, scalar, true);

    double elapsed_cpu = 0.0;
    double elapsed_memcpy = 0.0;
    double elapsed_wormhole = 0.0;
    // [CHANGED] was std::chrono::_V2::system_clock::time_point (libstdc++-internal name, not portable)
    std::chrono::high_resolution_clock::time_point start_total, end_total, start_wormhole, end_wormhole, start_memcpy, end_memcpy, start_cpu, end_cpu;
    std::chrono::duration<double, std::milli> elapsed;

    int lr_count = (cols-1) * sizeof(bfloat16);
    // int ud_count = (rows-1) * cols * sizeof(bfloat16);

    bfloat16* out = output.data()+cols;
    bfloat16* up_ptr = up.data();
    bfloat16* down_ptr = down.data();
    bfloat16* left_ptr = left.data();
    bfloat16* right_ptr = right.data();


    start_total = std::chrono::high_resolution_clock::now();

    for(int i = 0; i<iterations; i++){

        // NOTE: the first iteration also includes the JIT compilation of the kernels.
        start_wormhole = std::chrono::high_resolution_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, true);
        end_wormhole = std::chrono::high_resolution_clock::now();
        elapsed = end_wormhole - start_wormhole;
        elapsed_wormhole += elapsed.count();

        start_memcpy = std::chrono::high_resolution_clock::now();
        // [CHANGED] was EnqueueReadMeshBuffer(cq, output_dram_buffer, out, true): wrong argument order,
        // and the free function takes a std::vector (which it resizes), not a pointer.
        // The member function reads the whole buffer into a raw pointer; it must be blocking.
        cq.enqueue_read_mesh_buffer(out, output_dram_buffer, true);
        end_memcpy = std::chrono::high_resolution_clock::now();
        elapsed = end_memcpy - start_memcpy;
        elapsed_memcpy += elapsed.count();


        if (i != iterations-1){

            start_cpu = std::chrono::high_resolution_clock::now();
            out[(rows/2)*cols + cols/2] = 100.0f;
            // TOP: Copy rows 0 to rows-2 to out_top[cols, 2*cols, ..., (rows-1)*cols]
            // DOWN: Copy rows 1 to rows-1 to out_down[0, cols, ..., (rows-2)*cols]
            up_ptr = out - cols;
            down_ptr = out + cols;

            // LEFT and RIGHT: Copy cols 1 to cols-1 for each row
            for (int r = 0; r < rows; r++) {
                // LEFT: Copy cols 1 to cols-1 to out_left[r*cols + 1, ..., r*cols + cols-1]
                // RIGHT: Copy cols 1 to cols-1 to out_right[r*cols, ..., r*cols + cols-2]
                std::memcpy(left_ptr+(r*cols)+1, out+(r*cols), lr_count);
                std::memcpy(right_ptr+(r*cols), out+(r*cols)+1, lr_count);
            }

            end_cpu = std::chrono::high_resolution_clock::now();
            elapsed = end_cpu - start_cpu;
            elapsed_cpu += elapsed.count();

            start_memcpy = std::chrono::high_resolution_clock::now();
            // [CHANGED] raw-pointer writes use the MeshCommandQueue member function
            cq.enqueue_write_mesh_buffer(up_dram_buffer, up_ptr, true);
            cq.enqueue_write_mesh_buffer(left_dram_buffer, left_ptr, true);
            cq.enqueue_write_mesh_buffer(right_dram_buffer, right_ptr, true);
            cq.enqueue_write_mesh_buffer(down_dram_buffer, down_ptr, true);
            end_memcpy = std::chrono::high_resolution_clock::now();
            elapsed = end_memcpy - start_memcpy;
            elapsed_memcpy += elapsed.count();
        }
    }

    end_total = std::chrono::high_resolution_clock::now();
    elapsed = end_total - start_total;
    cout << "-TOTAL- " << elapsed.count() << " ms" << endl;
    cout << "-CPU- " << elapsed_cpu << " ms" << endl;
    cout << "-MEMCPY- " << elapsed_memcpy << " ms" << endl;
    cout << "-WORMHOLE- " << elapsed_wormhole << " ms" << endl;

    distributed::Finish(cq);

    return 0;
}


int main(int argc, char** argv) {


    uint32_t iterations;
    uint32_t rows;
    uint32_t cols;

    if (argc >= 4) {
        iterations = atoi(argv[1]);
        rows = atoi(argv[2]);
        cols = atoi(argv[3]);
        cout << "Using arguments: ITERATIONS=" << iterations << ", ROWS=" << rows << ", COLS=" << cols << endl;
    } else {
        cout << "Usage: " << argv[0] << " <iterations> <rows> <cols>" << endl;
        return -1;
    }

    //! To define by the input
    constexpr uint32_t stencil_order = 1;
    //! To define by the input

    //! direct from the input
    size_t buffer_size = rows * cols * sizeof(bfloat16);
    //* stencil
    // const uint32_t stencil_rows = TILE_HEIGHT;
    // const uint32_t stencil_cols = TILE_WIDTH;
    //* padding
    const uint32_t rows_pad = rows + stencil_order * 2;
    const uint32_t cols_pad = cols + stencil_order * 2;
    // const uint32_t padding_elements = 2*(rows_pad+cols_pad - 2);
    // const size_t pad_buffer_size = rows_pad * cols_pad * sizeof(bfloat16);
    //! direct from the input

    uint32_t num_tiles, dram_buffer_size;

    if (buffer_size < TILE_SIZE){
        cerr << "Error: problem size must be at least " << TILE_SIZE << " elements." << endl;
        return -1;
    }

    //* ----------
    //* INPUT ALLOCATION
    //* ----------

    vector<bfloat16> input_vec(rows * cols);
    for(int i = 0; i<rows * cols; i++){
        input_vec[i] = bfloat16(0.0f);
    }
    input_vec[(rows/2)*cols + cols/2] = 100.0f;

    //* ----------
    //* PADDING
    //* ----------

    // Pad the input, and the output but it's not necessary
    vector<bfloat16> input_vec_pad(rows_pad * cols_pad, 0.0f);
    vector<bfloat16> output_vec_pad(rows_pad * cols_pad, 0.0f);
    pad_with_zeros(input_vec.data(), input_vec_pad.data(), rows, cols, 1);

    //golden_stencil(input_vec_pad, output_vec_pad, rows_pad, cols_pad, iterations);

    //* ----------
    //* ALIGNEMENT
    //* ----------

    dram_buffer_size = rows * cols * sizeof(bfloat16);
    // dram_buffer_size = align_vector_size(input_vec_i2r, i2r_buffer_size, TILE_SIZE);
    // diff_dram = (dram_buffer_size - i2r_buffer_size) / sizeof(bfloat16);

    num_tiles = dram_buffer_size / TILE_SIZE;

    cout << "Problem shape: " << rows << "x" << cols << endl;
    cout << "Stencil order: " << stencil_order << endl;
    cout << "Padded shape: " << rows_pad << "x" << cols_pad << endl;
    cout << "DRAM buffer size (bytes): " << buffer_size << endl;
    cout << "Number of tiles: " << num_tiles << endl;

    //! KERNEL AREA
    int device_id = 0;
    // [CHANGED] CreateDevice(device_id) (deprecated) -> MeshDevice::create_unit_mesh(device_id).
    // A MeshDevice is a software concept that allows developers to virtualize a cluster of connected devices as a
    // single object, maintaining uniform memory and runtime state across all physical devices. A UnitMesh is a 1x1
    // MeshDevice that allows users to interface with a single physical device.
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    const auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] = split_work_to_cores(core_grid, num_tiles);

    vector<bfloat16> up_vec(rows*cols);
    vector<bfloat16> left_vec(rows*cols);
    vector<bfloat16> right_vec(rows*cols);
    vector<bfloat16> down_vec(rows*cols);
    vector<bfloat16> scalar_vec(TILE_WIDTH*TILE_HEIGHT*num_cores, 0.25f);
    vector<bfloat16> output_vec(dram_buffer_size/sizeof(bfloat16)+2*cols, 0.0f);

    extract_submats_5p(input_vec_pad.data(),
        up_vec.data(),
        left_vec.data(),
        right_vec.data(),
        down_vec.data(),
        rows,
        cols,
        cols_pad
    );

    axpy_ttker(
        up_vec,
        left_vec,
        right_vec,
        down_vec,
        scalar_vec,
        output_vec,
        num_tiles,
        rows,
        cols,
        iterations,
        mesh_device.get()  // [CHANGED] was: device
    );

    mesh_device->close();  // [CHANGED] was: CloseDevice(device)
    //! KERNEL AREA

    cout << "Output: " << endl;
    output_vec[((rows/2)*cols + cols/2)+cols] = 100.0f;
    printMat(output_vec.data() + cols, rows, cols);

    return 0;
}