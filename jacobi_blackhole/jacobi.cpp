// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// 5-point Jacobi solver for the Laplace equation, ported from jacobi_optimised (Grayskull e150) to
// Blackhole with tt-metal 1f2ae9b5.
//
// Usage: ./jacobi X Y ITS CX CY [--no-verify]
//   X, Y    interior grid size, multiples of 32
//   ITS     number of Jacobi sweeps
//   CX, CY  cores along the row (x) and column (y) dimension of the grid

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef JACOBI_KERNEL_DIR
#define JACOBI_KERNEL_DIR "kernels"
#endif

using namespace tt::tt_metal;

namespace {

constexpr uint32_t BATCH = 32;                                     // batch edge: one 32 x 32 page
constexpr uint32_t WINDOW = BATCH + 2;                             // batch plus a one-point halo
constexpr uint32_t TILE_BYTES = BATCH * BATCH * sizeof(bfloat16);  // 2 KB
constexpr uint32_t CB_PAGES = 4;                                   // pages per circular buffer

constexpr tt::CBIndex CB_EAST = tt::CBIndex::c_0;
constexpr tt::CBIndex CB_WEST = tt::CBIndex::c_1;
constexpr tt::CBIndex CB_SOUTH = tt::CBIndex::c_2;
constexpr tt::CBIndex CB_NORTH = tt::CBIndex::c_3;
constexpr tt::CBIndex CB_WINDOW = tt::CBIndex::c_4;
constexpr tt::CBIndex CB_OUT = tt::CBIndex::c_16;

// Per-core semaphores, each counting finished sweeps: of this core's writer, and of each neighbour.
enum Semaphore : uint32_t { SEM_SELF, SEM_FROM_NORTH, SEM_FROM_SOUTH, SEM_FROM_WEST, SEM_FROM_EAST, NUM_SEMAPHORES };
constexpr uint32_t HAS_NORTH = 1, HAS_SOUTH = 2, HAS_WEST = 4, HAS_EAST = 8;

struct Config {
    uint32_t x = 0, y = 0, its = 0, cx = 0, cy = 0;
    bool verify = true;
};

// Padded grid in DRAM: one halo row above and below the interior, `pad` columns on each side.
struct GridLayout {
    uint32_t align = 0;         // DRAM read alignment in bytes (64 on Blackhole)
    uint32_t pad = 0;           // padding columns per side, align / 2, so interior rows start aligned
    uint32_t rows = 0;          // X + 2
    uint32_t cols = 0;          // Y + 2 * pad
    uint32_t row_bytes = 0;     // one DRAM page per padded row
    uint32_t window_bytes = 0;  // L1 for one 34-row window, as laid out by stream_in
};

GridLayout make_layout(const Config& cfg, uint32_t align) {
    GridLayout g;
    g.align = align;
    g.pad = align / sizeof(bfloat16);
    g.rows = cfg.x + 2;
    g.cols = cfg.y + 2 * g.pad;
    g.row_bytes = g.cols * sizeof(bfloat16);
    const uint32_t line_bytes = (align + WINDOW * sizeof(bfloat16) + align - 1) / align * align;
    g.window_bytes = WINDOW * line_bytes;
    return g;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::string kernel_path(const char* name) {
    const char* dir = std::getenv("JACOBI_KERNEL_DIR");
    return std::string(dir != nullptr ? dir : JACOBI_KERNEL_DIR) + "/" + name;
}

bool parse_uint(const char* text, uint32_t& value) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed > UINT32_MAX) {
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_args(int argc, char** argv, Config& cfg) {
    std::vector<const char*> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-verify") {
            cfg.verify = false;
        } else {
            positional.push_back(argv[i]);
        }
    }
    uint32_t* fields[] = {&cfg.x, &cfg.y, &cfg.its, &cfg.cx, &cfg.cy};
    bool ok = positional.size() == 5;
    for (size_t i = 0; ok && i < 5; ++i) {
        ok = parse_uint(positional[i], *fields[i]);
    }
    if (!ok) {
        fprintf(stderr, "Usage: %s X Y ITS CX CY [--no-verify]\n", argv[0]);
        fprintf(stderr, "  X, Y: interior grid size; ITS: sweeps; CX, CY: cores along x (rows) and y (columns)\n");
        return false;
    }
    if (cfg.x < BATCH || cfg.x % BATCH != 0 || cfg.y < BATCH || cfg.y % BATCH != 0) {
        fprintf(stderr, "X and Y must be multiples of 32 and at least 32\n");
        return false;
    }
    if (cfg.its == 0 || cfg.cx == 0 || cfg.cy == 0) {
        fprintf(stderr, "ITS, CX and CY must be at least 1\n");
        return false;
    }
    if (cfg.cx > cfg.x / BATCH || cfg.cy > cfg.y / BATCH) {
        fprintf(stderr, "Every core needs at least one 32 x 32 batch: CX <= X / 32 and CY <= Y / 32\n");
        return false;
    }
    return true;
}

// Block split used for both dimensions: n batches over c cores, the first n % c ranks get one more.
// Returns {first batch, number of batches} for `rank`.
std::pair<uint32_t, uint32_t> split(uint32_t n, uint32_t c, uint32_t rank) {
    const uint32_t base = n / c;
    const uint32_t extra = n % c;
    return {rank * base + std::min(rank, extra), base + (rank < extra ? 1 : 0)};
}

// Zero interior and halo rows, 1.0 in the last padding column on the left and 20.0 in the first
// padding column on the right of every row.
std::vector<bfloat16> initial_grid(const GridLayout& g) {
    std::vector<bfloat16> grid(static_cast<size_t>(g.rows) * g.cols, bfloat16(0.0f));
    for (uint32_t r = 0; r < g.rows; ++r) {
        grid[static_cast<size_t>(r) * g.cols + g.pad - 1] = bfloat16(1.0f);
        grid[static_cast<size_t>(r) * g.cols + g.cols - g.pad] = bfloat16(20.0f);
    }
    return grid;
}

// CPU reference: the same sweeps, summed in the device's order (E + W + S + N) and rounded to bf16
// once per sweep, as the device stores it.
std::vector<float> golden_jacobi(const std::vector<bfloat16>& init, const GridLayout& g, uint32_t its) {
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
                const float sum = ((mid[c + 1] + mid[c - 1]) + down[c]) + up[c];
                out[c] = static_cast<float>(bfloat16(0.25f * sum));
            }
        }
        std::swap(u, v);
    }
    return u;
}

// Compare every point, padding and halo included. The device keeps partial sums in fp32 but can round
// them differently from the CPU, so allow 1e-3 absolute plus 2^-6 relative (2 to 4 bf16 ulps).
bool compare(const std::vector<bfloat16>& device, const std::vector<float>& reference, const GridLayout& g) {
    size_t mismatches = 0;
    size_t exact = 0;
    float max_diff = 0.0f;
    for (uint32_t r = 0; r < g.rows; ++r) {
        for (uint32_t c = 0; c < g.cols; ++c) {
            const size_t i = static_cast<size_t>(r) * g.cols + c;
            const float got = static_cast<float>(device[i]);
            const float want = reference[i];
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
    const size_t total = device.size();
    printf(
        "%s: %zu of %zu grid points within tolerance (%zu identical), max abs difference %g\n",
        mismatches == 0 ? "Match" : "Error",
        total - mismatches,
        total,
        exact,
        static_cast<double>(max_diff));
    return mismatches == 0;
}

// One program over the CX x CY core range: every core runs `its` sweeps over its own block of
// batches, synchronising with its four neighbours between sweeps.
distributed::MeshWorkload build_workload(
    distributed::MeshDevice& mesh,
    const Config& cfg,
    const GridLayout& g,
    const distributed::MeshBuffer& grid_a,
    const distributed::MeshBuffer& grid_b,
    uint32_t its) {
    Program program = CreateProgram();
    const tt::tt_metal::CoreRange cores({0, 0}, {cfg.cx - 1, cfg.cy - 1});

    auto make_cb = [&](tt::CBIndex index, uint32_t pages, uint32_t page_bytes) {
        const CircularBufferConfig config =
            CircularBufferConfig(pages * page_bytes, {{index, tt::DataFormat::Float16_b}}).set_page_size(index, page_bytes);
        CreateCircularBuffer(program, cores, config);
    };
    for (tt::CBIndex index : {CB_EAST, CB_WEST, CB_SOUTH, CB_NORTH, CB_OUT}) {
        make_cb(index, CB_PAGES, TILE_BYTES);
    }
    // Reader scratch for two windows, plus slack to align its base to the DRAM alignment.
    make_cb(CB_WINDOW, 1, 2 * g.window_bytes + g.align);

    std::array<uint32_t, NUM_SEMAPHORES> semaphores{};
    for (uint32_t& id : semaphores) {
        id = CreateSemaphore(program, cores, 0);
    }

    // Dataflow compile-time args: the DRAM alignment, then the grid layout (A and B share it).
    std::vector<uint32_t> dataflow_ct_args = {g.align};
    TensorAccessorArgs(grid_a).append_to(dataflow_ct_args);

    const KernelHandle reader = CreateKernel(
        program,
        kernel_path("dataflow/stream_in.cpp"),
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = dataflow_ct_args});
    const KernelHandle writer = CreateKernel(
        program,
        kernel_path("dataflow/stream_out.cpp"),
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = dataflow_ct_args});
    const KernelHandle compute = CreateKernel(
        program,
        kernel_path("compute/laplace.cpp"),
        cores,
        ComputeConfig{.math_fidelity = tt::tt_metal::MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});

    const uint32_t a_addr = static_cast<uint32_t>(grid_a.address());
    const uint32_t b_addr = static_cast<uint32_t>(grid_b.address());
    for (uint32_t rx = 0; rx < cfg.cx; ++rx) {
        for (uint32_t ry = 0; ry < cfg.cy; ++ry) {
            const auto [bx0, nbx] = split(cfg.x / BATCH, cfg.cx, rx);
            const auto [by0, nby] = split(cfg.y / BATCH, cfg.cy, ry);

            // Neighbouring blocks: north/south along x (rows), west/east along y (columns).
            uint32_t neighbours = 0;
            std::array<CoreCoord, 4> noc{};  // NoC coordinates of north, south, west, east
            auto add_neighbour = [&](uint32_t bit, size_t slot, uint32_t nx, uint32_t ny) {
                neighbours |= bit;
                noc[slot] = mesh.worker_core_from_logical_core(CoreCoord{nx, ny});
            };
            if (rx > 0) {
                add_neighbour(HAS_NORTH, 0, rx - 1, ry);
            }
            if (rx + 1 < cfg.cx) {
                add_neighbour(HAS_SOUTH, 1, rx + 1, ry);
            }
            if (ry > 0) {
                add_neighbour(HAS_WEST, 2, rx, ry - 1);
            }
            if (ry + 1 < cfg.cy) {
                add_neighbour(HAS_EAST, 3, rx, ry + 1);
            }

            std::vector<uint32_t> dataflow_args = {
                a_addr,
                b_addr,
                bx0,
                nbx,
                by0,
                nby,
                its,
                semaphores[SEM_SELF],
                semaphores[SEM_FROM_NORTH],
                semaphores[SEM_FROM_SOUTH],
                semaphores[SEM_FROM_WEST],
                semaphores[SEM_FROM_EAST],
                neighbours};
            for (const CoreCoord& c : noc) {
                dataflow_args.push_back(static_cast<uint32_t>(c.x));
                dataflow_args.push_back(static_cast<uint32_t>(c.y));
            }

            const CoreCoord core{rx, ry};
            SetRuntimeArgs(program, reader, core, dataflow_args);
            SetRuntimeArgs(program, writer, core, dataflow_args);
            SetRuntimeArgs(program, compute, core, {nbx * nby * its});
        }
    }

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh.shape()), std::move(program));
    return workload;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        return 1;
    }

    std::shared_ptr<distributed::MeshDevice> mesh = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh->mesh_command_queue();
    const CoreCoord grid_size = mesh->compute_with_storage_grid_size();
    printf(
        "Executing over x=%u, y=%u cores, there are a total of x=%zu y=%zu cores available\n",
        cfg.cx,
        cfg.cy,
        grid_size.x,
        grid_size.y);
    if (cfg.cx > grid_size.x || cfg.cy > grid_size.y) {
        fprintf(stderr, "Requested %u x %u cores, but the compute grid is %zu x %zu\n", cfg.cx, cfg.cy, grid_size.x, grid_size.y);
        mesh->close();
        return 1;
    }

    const GridLayout g = make_layout(cfg, hal::get_dram_alignment());
    printf("Running Jacobi for x=%u, y=%u over %u iterations\n", cfg.x, cfg.y, cfg.its);
    printf("DRAM alignment %u B: %u padding columns per side, %u B per grid row\n", g.align, g.pad, g.row_bytes);

    // Two DRAM grids, written alternately. One page per padded row spreads the rows over all banks.
    const distributed::DeviceLocalBufferConfig dram_config{.page_size = g.row_bytes, .buffer_type = BufferType::DRAM};
    const distributed::ReplicatedBufferConfig grid_config{.size = static_cast<uint64_t>(g.rows) * g.row_bytes};
    std::shared_ptr<distributed::MeshBuffer> grid_a = distributed::MeshBuffer::create(grid_config, dram_config, mesh.get());
    std::shared_ptr<distributed::MeshBuffer> grid_b = distributed::MeshBuffer::create(grid_config, dram_config, mesh.get());

    const std::vector<bfloat16> init = initial_grid(g);
    std::vector<float> reference;
    if (cfg.verify) {
        const auto start = std::chrono::steady_clock::now();
        reference = golden_jacobi(init, g, cfg.its);
        printf("Calculated reference version over %u iterations in %.4f secs\n", cfg.its, seconds_since(start));
    }

    // A one-sweep run compiles the kernels, so the timed run below measures execution only.
    {
        distributed::MeshWorkload warmup = build_workload(*mesh, cfg, g, *grid_a, *grid_b, 1);
        distributed::EnqueueMeshWorkload(cq, warmup, true);
    }
    distributed::MeshWorkload workload = build_workload(*mesh, cfg, g, *grid_a, *grid_b, cfg.its);

    printf("--- LAUNCH ---\n");
    auto start = std::chrono::steady_clock::now();
    distributed::EnqueueWriteMeshBuffer(cq, grid_a, init, false);
    distributed::EnqueueWriteMeshBuffer(cq, grid_b, init, false);
    distributed::Finish(cq);
    const double xfer_on_time = seconds_since(start);

    start = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, true);
    const double exec_time = seconds_since(start);
    printf("--- COMPLETED RUN ---\n");

    // The last sweep (ITS - 1) wrote B if it was even, A if it was odd.
    start = std::chrono::steady_clock::now();
    std::vector<bfloat16> result;
    distributed::EnqueueReadMeshBuffer(cq, result, cfg.its % 2 == 1 ? grid_b : grid_a, true);
    const double xfer_off_time = seconds_since(start);
    printf("--- COPIED DATA BACK ---\n");

    bool ok = true;
    if (cfg.verify) {
        ok = compare(result, reference, g);
    }
    mesh->close();

    const double total_time = xfer_on_time + exec_time + xfer_off_time;
    printf(
        "Run completed, total time %.4f sec. %.4f sec transfer on, %.4f sec execution, %.4f sec transfer off\n",
        total_time,
        xfer_on_time,
        exec_time,
        xfer_off_time);
    const uint64_t total_points = static_cast<uint64_t>(cfg.x) * cfg.y * cfg.its;
    printf("Total points %llu\n", static_cast<unsigned long long>(total_points));
    printf("Execution GPts: %.4f\n", static_cast<double>(total_points) / exec_time / (1024.0 * 1024.0 * 1024.0));
    printf("Total GPts: %.4f\n", static_cast<double>(total_points) / total_time / (1024.0 * 1024.0 * 1024.0));
    return ok ? 0 : 1;
}
