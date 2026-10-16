// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// 5-point Jacobi stencil as an AXPY: every sweep the host builds the UP, LEFT, RIGHT and DOWN shifted
// copies of the grid, and the device computes 0.25 * (UP + DOWN + LEFT + RIGHT) tile by tile.
//
// Usage: ./axpy_stencil ITERATIONS ROWS COLS [CX CY]
//   ROWS x COLS  interior grid size, ROWS * COLS a multiple of 1024 (one tile)
//   CX, CY       cores along x and y (default: the whole compute grid of the device, 11 x 10 on Blackhole)
//
// The problem is the one of optimized_blackhole (zero top and bottom halo, 1.0 on the left and 20.0
// on the right), so every run also times the same CPU versions: the single-threaded reference that
// mirrors the device's bf16 rounding (and checks its result), and an fp32 version on CPU_THREADS threads.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "submatrices.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace std;

#define TILE_WIDTH 32 // bfloats
#define TILE_HEIGHT 32 // bfloats
#define TILE_SIZE 2048 // bytes

// I wanto to have all the SRAM available, maybe I could use also multiple CBs
#define SRAM_TILES 64

constexpr uint32_t CPU_THREADS = 32;  // threads of golden_optimized

struct Timings {
    double total = 0.0;     // wall time of the whole timed region
    double xfer_on = 0.0;   // host-to-device writes (initial and per sweep)
    double exec = 0.0;      // device execution
    double xfer_off = 0.0;  // device-to-host reads
    double host = 0.0;      // CPU work building the shifted copies between sweeps
};

static double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

//* ---------------------------------------------------------
//* CPU VERSIONS (from optimized_blackhole/main.cpp)
//* ---------------------------------------------------------

// Padded grid: one halo row above and below the interior, `pad` columns on each side.
struct GridLayout {
    uint32_t pad = 1;
    uint32_t rows = 0;  // ROWS + 2
    uint32_t cols = 0;  // COLS + 2 * pad
};

// Block split: n items over c workers, the first n % c ranks get one more.
// Returns {first item, number of items} for `rank`.
static std::pair<uint32_t, uint32_t> split_rows(uint32_t n, uint32_t c, uint32_t rank) {
    const uint32_t base = n / c;
    const uint32_t extra = n % c;
    return {rank * base + std::min(rank, extra), base + (rank < extra ? 1 : 0)};
}

// Zero interior and halo rows, 1.0 in the last padding column on the left and 20.0 in the first
// padding column on the right of every row.
static std::vector<bfloat16> initial_grid(const GridLayout& g) {
    std::vector<bfloat16> grid(static_cast<size_t>(g.rows) * g.cols, bfloat16(0.0f));
    for (uint32_t r = 0; r < g.rows; ++r) {
        grid[static_cast<size_t>(r) * g.cols + g.pad - 1] = bfloat16(1.0f);
        grid[static_cast<size_t>(r) * g.cols + g.cols - g.pad] = bfloat16(20.0f);
    }
    return grid;
}

// CPU reference: the same sweeps, summed in the device's order ((UP + DOWN) + LEFT) + RIGHT, rounded
// to bf16 after every operation as the compute kernel packs every partial sum to a bf16 tile.
static std::vector<float> golden_jacobi(const std::vector<bfloat16>& init, const GridLayout& g, uint32_t its) {
    auto bf16 = [](float x) { return static_cast<float>(bfloat16(x)); };
    std::vector<float> u(init.size());
    for (size_t i = 0; i < init.size(); ++i) {
        u[i] = static_cast<float>(init[i]);
    }
    std::vector<float> v = u;
    for (uint32_t it = 0; it < its; ++it) {
        for (uint32_t r = 1; r + 1 < g.rows; ++r) {
            const float* up = &u[static_cast<size_t>(r - 1) * g.cols];
            const float* mid = &u[static_cast<size_t>(r) * g.cols];
            const float* down = &u[static_cast<size_t>(r + 1) * g.cols];
            float* out = &v[static_cast<size_t>(r) * g.cols];
            for (uint32_t c = g.pad; c < g.cols - g.pad; ++c) {
                float sum = bf16(up[c] + down[c]);
                sum = bf16(mid[c - 1] + sum);
                sum = bf16(mid[c + 1] + sum);
                out[c] = bf16(0.25f * sum);
            }
        }
        std::swap(u, v);
    }
    return u;
}

// Optimised CPU baseline: the same sweeps in plain fp32 (no bf16 rounding) on `num_threads` threads,
// each owning a block of interior rows, with a barrier between sweeps. Every thread first touches its
// own rows so they land in its NUMA node. Returns the seconds spent in the sweeps only, the
// counterpart of the device's execution time (allocation, initialisation and thread start-up excluded).
static double golden_optimized(const GridLayout& g, uint32_t its, uint32_t num_threads) {
    const size_t cols = g.cols;
    std::unique_ptr<float[]> grid_a(new float[g.rows * cols]);  // uninitialised, see first touch above
    std::unique_ptr<float[]> grid_b(new float[g.rows * cols]);
    const uint32_t interior = g.rows - 2;
    num_threads = std::min(num_threads, interior);

    std::barrier sweep_done(num_threads);
    std::barrier start_stop(num_threads + 1);  // the calling thread joins to time the sweeps
    auto worker = [&](uint32_t rank) {
        const auto [first, count] = split_rows(interior, num_threads, rank);
        // Interior rows are 1 .. ROWS; thread 0 also owns the top halo row, the last one the bottom halo row.
        const uint32_t r0 = first + 1, r1 = first + 1 + count;
        const uint32_t init0 = rank == 0 ? 0 : r0;
        const uint32_t init1 = rank == num_threads - 1 ? g.rows : r1;
        for (float* grid : {grid_a.get(), grid_b.get()}) {
            for (uint32_t r = init0; r < init1; ++r) {
                float* row = grid + r * cols;
                std::fill(row, row + cols, 0.0f);
                row[g.pad - 1] = 1.0f;  // same boundary as initial_grid
                row[cols - g.pad] = 20.0f;
            }
        }
        float* src = grid_a.get();
        float* dst = grid_b.get();
        start_stop.arrive_and_wait();
        for (uint32_t it = 0; it < its; ++it) {
            for (uint32_t r = r0; r < r1; ++r) {
                const float* __restrict up = src + (r - 1) * cols;
                const float* __restrict mid = src + r * cols;
                const float* __restrict down = src + (r + 1) * cols;
                float* __restrict out = dst + r * cols;
                for (size_t c = g.pad; c < cols - g.pad; ++c) {
                    out[c] = 0.25f * (mid[c + 1] + mid[c - 1] + down[c] + up[c]);
                }
            }
            sweep_done.arrive_and_wait();
            std::swap(src, dst);
        }
        start_stop.arrive_and_wait();
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (uint32_t t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    start_stop.arrive_and_wait();
    const auto start = std::chrono::steady_clock::now();
    start_stop.arrive_and_wait();
    const double elapsed = seconds_since(start);
    for (std::thread& t : threads) {
        t.join();
    }
    return elapsed;
}

// Compare every interior point (`device` is ROWS x COLS, `reference` the padded grid). Allow 1e-3
// absolute plus 2^-6 relative (2 to 4 bf16 ulps), as optimized_blackhole does.
static bool compare_grids(const bfloat16* device, const std::vector<float>& reference, const GridLayout& g) {
    const uint32_t rows = g.rows - 2, cols = g.cols - 2 * g.pad;
    size_t mismatches = 0;
    size_t exact = 0;
    float max_diff = 0.0f;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const float got = static_cast<float>(device[static_cast<size_t>(r) * cols + c]);
            const float want = reference[static_cast<size_t>(r + 1) * g.cols + c + g.pad];
            const float diff = std::fabs(got - want);
            exact += diff == 0.0f ? 1 : 0;
            max_diff = std::max(max_diff, diff);
            if (!(diff <= 1e-3f + std::fabs(want) / 64.0f)) {  // also catches NaN
                if (mismatches < 10) {
                    printf("  mismatch at row %u, column %u: device %f, reference %f\n", r, c, got, want);
                }
                ++mismatches;
            }
        }
    }
    const size_t total = static_cast<size_t>(rows) * cols;
    printf(
        "%s: %zu of %zu grid points within tolerance (%zu identical), max abs difference %g\n",
        mismatches == 0 ? "Match" : "Error",
        total - mismatches,
        total,
        exact,
        static_cast<double>(max_diff));
    return mismatches == 0;
}

//* ---------------------------------------------------------
//* DEVICE VERSION
//* ---------------------------------------------------------

Timings axpy_ttker(
    vector<bfloat16>& up,
    vector<bfloat16>& left,
    vector<bfloat16>& right,
    vector<bfloat16>& down,
    vector<bfloat16>& output,
    uint32_t n_tiles,
    uint32_t rows,
    uint32_t cols,
    uint32_t iterations,
    CoreCoord core_grid,  // CX x CY cores to split the tiles over
    distributed::MeshDevice* mesh_device
) {

    //* ---------------------------------------------------------
    //* HOST INITIALIZATION
    //* ---------------------------------------------------------

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
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;

    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] = split_work_to_cores(core_grid, n_tiles);

    printf("Using %u of %zu x %zu cores\n", num_cores, core_grid.x, core_grid.y);
    cout << " Tensixes group 1 has " << work_per_core1 << " tiles" << endl;
    cout << " Tensixes group 2 has " << work_per_core2 << " tiles" << endl;

    // ---------------------------------------------------------
    // DRAM BUFFER CREATION: OFFCHIP GDDR6 MEMORY
    // ---------------------------------------------------------

    cout << "Creating DRAM buffers..." << endl;

    // One tile of 0.25 per core: each core reads its own, so they don't all hit the same DRAM page.
    vector<bfloat16> scalar(TILE_WIDTH * TILE_HEIGHT * num_cores, bfloat16(0.25f));

    // DeviceLocalBufferConfig: page_size and buffer_type; ReplicatedBufferConfig: size in bytes, per device.
    // DRAM buffers created this way are interleaved.
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = TILE_SIZE,
        .buffer_type = BufferType::DRAM};

    distributed::ReplicatedBufferConfig dram_inout_config{
        .size = up.size() * sizeof(bfloat16)};

    distributed::ReplicatedBufferConfig dram_scalar_config{
        .size = scalar.size() * sizeof(bfloat16)};

    std::shared_ptr<distributed::MeshBuffer> up_dram_buffer = distributed::MeshBuffer::create(dram_inout_config, dram_config, mesh_device);
    std::shared_ptr<distributed::MeshBuffer> left_dram_buffer = distributed::MeshBuffer::create(dram_inout_config, dram_config, mesh_device);
    std::shared_ptr<distributed::MeshBuffer> right_dram_buffer = distributed::MeshBuffer::create(dram_inout_config, dram_config, mesh_device);
    std::shared_ptr<distributed::MeshBuffer> down_dram_buffer = distributed::MeshBuffer::create(dram_inout_config, dram_config, mesh_device);
    std::shared_ptr<distributed::MeshBuffer> output_dram_buffer = distributed::MeshBuffer::create(dram_inout_config, dram_config, mesh_device);
    std::shared_ptr<distributed::MeshBuffer> scalar_dram_buffer = distributed::MeshBuffer::create(dram_scalar_config, dram_config, mesh_device);

    // ---------------------------------------------------------
    // SRAM BUFFER CREATION
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

    // ---------------------------------------------------------
    // SETUP RUNTIME ARGS: Set the runtime arguments for the kernels
    // ---------------------------------------------------------

    cout << "Setting up runtime arguments..." << endl;

    // MeshBuffer::address() returns a 64-bit DeviceAddr, but runtime args are uint32_t.
    const uint32_t up_addr     = static_cast<uint32_t>(up_dram_buffer->address());
    const uint32_t left_addr   = static_cast<uint32_t>(left_dram_buffer->address());
    const uint32_t right_addr  = static_cast<uint32_t>(right_dram_buffer->address());
    const uint32_t down_addr   = static_cast<uint32_t>(down_dram_buffer->address());
    const uint32_t scalar_addr = static_cast<uint32_t>(scalar_dram_buffer->address());
    const uint32_t output_addr = static_cast<uint32_t>(output_dram_buffer->address());

    uint32_t start_tile_idx = 0;
    uint32_t core_id = 0;  // index of the core's scalar tile, 0 .. num_cores - 1
    for (const auto& [group, work_per_core] : {std::pair{core_group_1, work_per_core1}, std::pair{core_group_2, work_per_core2}}) {
        for(const auto& core_range : group.ranges()){
            for(const auto& core : core_range) {
                tt_metal::SetRuntimeArgs( program, reader_kernel_id, core, {
                    up_addr,
                    left_addr,
                    right_addr,
                    down_addr,
                    scalar_addr,
                    start_tile_idx,
                    core_id,
                    work_per_core
                });

                tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, {
                    output_addr, start_tile_idx, work_per_core
                });

                tt_metal::SetRuntimeArgs(program, stencil_kernel_id, core, {
                    work_per_core
                });

                start_tile_idx += work_per_core;
                ++core_id;
            }
        }
    }

    // The program is added to the workload once; the same workload is enqueued every sweep.
    workload.add_program(device_range, std::move(program));

    // A first run compiles the kernels, so the timed sweeps below measure execution only. Its result
    // is overwritten by the first timed sweep.
    distributed::EnqueueMeshWorkload(cq, workload, true);

    // ---------------------------------------------------------
    // TIMED SWEEPS
    // ---------------------------------------------------------

    cout << "--- LAUNCH ---" << endl;

    Timings t;
    const auto start_total = std::chrono::steady_clock::now();

    auto start = std::chrono::steady_clock::now();
    distributed::EnqueueWriteMeshBuffer(cq, up_dram_buffer, up, true);
    distributed::EnqueueWriteMeshBuffer(cq, left_dram_buffer, left, true);
    distributed::EnqueueWriteMeshBuffer(cq, right_dram_buffer, right, true);
    distributed::EnqueueWriteMeshBuffer(cq, down_dram_buffer, down, true);
    distributed::EnqueueWriteMeshBuffer(cq, scalar_dram_buffer, scalar, true);
    t.xfer_on += seconds_since(start);

    const size_t lr_count = (cols-1) * sizeof(bfloat16);

    // `output` has one zero row above and below the grid, so UP and DOWN are views of it shifted by a row.
    bfloat16* out = output.data()+cols;
    bfloat16* up_ptr = out - cols;
    bfloat16* down_ptr = out + cols;
    bfloat16* left_ptr = left.data();
    bfloat16* right_ptr = right.data();

    for(uint32_t i = 0; i<iterations; i++){

        start = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, true);
        t.exec += seconds_since(start);

        start = std::chrono::steady_clock::now();
        // The member function reads the whole buffer into a raw pointer; it must be blocking.
        cq.enqueue_read_mesh_buffer(out, output_dram_buffer, true);
        t.xfer_off += seconds_since(start);

        if (i != iterations-1){

            start = std::chrono::steady_clock::now();
            // LEFT and RIGHT: shift every row by one column; column 0 of LEFT (1.0) and the last
            // column of RIGHT (20.0) keep the boundary.
            for (uint32_t r = 0; r < rows; r++) {
                std::memcpy(left_ptr+(r*cols)+1, out+(r*cols), lr_count);
                std::memcpy(right_ptr+(r*cols), out+(r*cols)+1, lr_count);
            }
            t.host += seconds_since(start);

            start = std::chrono::steady_clock::now();
            cq.enqueue_write_mesh_buffer(up_dram_buffer, up_ptr, true);
            cq.enqueue_write_mesh_buffer(left_dram_buffer, left_ptr, true);
            cq.enqueue_write_mesh_buffer(right_dram_buffer, right_ptr, true);
            cq.enqueue_write_mesh_buffer(down_dram_buffer, down_ptr, true);
            t.xfer_on += seconds_since(start);
        }
    }

    distributed::Finish(cq);
    t.total = seconds_since(start_total);
    cout << "--- COMPLETED RUN ---" << endl;

    return t;
}


int main(int argc, char** argv) {

    uint32_t iterations;
    uint32_t rows;
    uint32_t cols;
    uint32_t cx = 0, cy = 0;  // 0: the whole compute grid

    if (argc == 4 || argc == 6) {
        iterations = atoi(argv[1]);
        rows = atoi(argv[2]);
        cols = atoi(argv[3]);
        if (argc == 6) {
            cx = atoi(argv[4]);
            cy = atoi(argv[5]);
        }
        cout << "Using arguments: ITERATIONS=" << iterations << ", ROWS=" << rows << ", COLS=" << cols << endl;
    } else {
        cout << "Usage: " << argv[0] << " <iterations> <rows> <cols> [<cx> <cy>]" << endl;
        return -1;
    }

    constexpr uint32_t stencil_order = 1;

    const size_t elements = static_cast<size_t>(rows) * cols;
    if (iterations == 0 || rows < 2 || cols < 2 || elements % (TILE_WIDTH * TILE_HEIGHT) != 0) {
        cerr << "Error: ITERATIONS must be at least 1, and ROWS * COLS a multiple of " << TILE_WIDTH * TILE_HEIGHT << "." << endl;
        return -1;
    }
    const uint32_t num_tiles = elements / (TILE_WIDTH * TILE_HEIGHT);

    const GridLayout g{.pad = stencil_order, .rows = rows + 2 * stencil_order, .cols = cols + 2 * stencil_order};

    cout << "Problem shape: " << rows << "x" << cols << endl;
    cout << "Stencil order: " << stencil_order << endl;
    cout << "Padded shape: " << g.rows << "x" << g.cols << endl;
    cout << "DRAM buffer size (bytes): " << elements * sizeof(bfloat16) << endl;
    cout << "Number of tiles: " << num_tiles << endl;

    //* ----------
    //* INPUT AND CPU VERSIONS
    //* ----------

    // Zero grid in a zero halo, with the 1.0 / 20.0 left / right boundary of optimized_blackhole.
    vector<bfloat16> input_vec_pad = initial_grid(g);

    const auto golden_start = std::chrono::steady_clock::now();
    const std::vector<float> reference = golden_jacobi(input_vec_pad, g, iterations);
    printf("Calculated reference version over %u iterations in %.4f secs\n", iterations, seconds_since(golden_start));
    const uint32_t cpu_threads = std::min(CPU_THREADS, rows);
    const double golden_opt_time = golden_optimized(g, iterations, cpu_threads);
    printf(
        "Calculated optimized fp32 version over %u iterations on %u threads in %.4f secs\n",
        iterations,
        cpu_threads,
        golden_opt_time);

    //! KERNEL AREA
    int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    const CoreCoord device_grid = mesh_device->compute_with_storage_grid_size();
    if (cx == 0 && cy == 0) {
        cx = device_grid.x;
        cy = device_grid.y;
    }
    printf(
        "Executing over x=%u, y=%u cores, there are a total of x=%zu y=%zu cores available\n",
        cx,
        cy,
        device_grid.x,
        device_grid.y);
    if (cx == 0 || cy == 0 || cx > device_grid.x || cy > device_grid.y) {
        cerr << "Error: requested " << cx << " x " << cy << " cores, but the compute grid is "
             << device_grid.x << " x " << device_grid.y << "." << endl;
        mesh_device->close();
        return -1;
    }

    vector<bfloat16> up_vec(elements);
    vector<bfloat16> left_vec(elements);
    vector<bfloat16> right_vec(elements);
    vector<bfloat16> down_vec(elements);
    vector<bfloat16> output_vec(elements + 2 * cols, 0.0f);  // plus a zero row above and below

    extract_submats_5p(input_vec_pad.data(),
        up_vec.data(),
        left_vec.data(),
        right_vec.data(),
        down_vec.data(),
        rows,
        cols,
        g.cols
    );
    input_vec_pad = vector<bfloat16>();  // no longer needed: free it before the device run

    const Timings t = axpy_ttker(
        up_vec,
        left_vec,
        right_vec,
        down_vec,
        output_vec,
        num_tiles,
        rows,
        cols,
        iterations,
        CoreCoord(cx, cy),
        mesh_device.get()
    );

    mesh_device->close();
    //! KERNEL AREA

    const bool ok = compare_grids(output_vec.data() + cols, reference, g);

    printf(
        "Run completed, total time %.4f sec. %.4f sec transfer on, %.4f sec execution, %.4f sec transfer off, "
        "%.4f sec host\n",
        t.total,
        t.xfer_on,
        t.exec,
        t.xfer_off,
        t.host);
    const uint64_t total_points = static_cast<uint64_t>(elements) * iterations;
    printf("Total points %llu\n", static_cast<unsigned long long>(total_points));
    printf("Execution GPts: %.4f\n", static_cast<double>(total_points) / t.exec / (1024.0 * 1024.0 * 1024.0));
    printf("Total GPts: %.4f\n", static_cast<double>(total_points) / t.total / (1024.0 * 1024.0 * 1024.0));

    return ok ? 0 : 1;
}
