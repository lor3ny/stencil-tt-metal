// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer (RISCV_0, NoC 0). For every sweep and every batch this core owns, it writes the 32 x 32
// result page into the grid that sweep writes (B on even sweeps, A on odd ones). At the end of each
// sweep it tells this core's reader, and the readers of the four neighbouring cores, that the sweep
// has landed in DRAM.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t BATCH = 32;                           // batch edge in grid points
constexpr uint32_t ELEM_BYTES = 2;                       // bf16
constexpr uint32_t PAGE_ROW_BYTES = BATCH * ELEM_BYTES;  // one row of a 32 x 32 page

// DRAM alignment from the host HAL; the host pads every grid row with ALIGN / 2 columns per side.
constexpr uint32_t ALIGN = get_compile_time_arg_val(0);
constexpr uint32_t PAD = ALIGN / ELEM_BYTES;

constexpr uint32_t cb_out = tt::CBIndex::c_16;

constexpr uint32_t HAS_NORTH = 1, HAS_SOUTH = 2, HAS_WEST = 4, HAS_EAST = 8;

}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t grid_a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grid_b_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t bx0 = get_arg_val<uint32_t>(arg++);
    const uint32_t nbx = get_arg_val<uint32_t>(arg++);
    const uint32_t by0 = get_arg_val<uint32_t>(arg++);
    const uint32_t nby = get_arg_val<uint32_t>(arg++);
    const uint32_t num_its = get_arg_val<uint32_t>(arg++);
    // Semaphores are allocated at the same L1 address on every core, so this core's addresses also
    // name the neighbours' copies.
    const uint32_t sem_self = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t sem_from_north = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t sem_from_south = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t sem_from_west = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t sem_from_east = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t neighbours = get_arg_val<uint32_t>(arg++);
    // NoC coordinates of the north, south, west and east neighbours (unused when absent).
    const uint32_t north_x = get_arg_val<uint32_t>(arg++);
    const uint32_t north_y = get_arg_val<uint32_t>(arg++);
    const uint32_t south_x = get_arg_val<uint32_t>(arg++);
    const uint32_t south_y = get_arg_val<uint32_t>(arg++);
    const uint32_t west_x = get_arg_val<uint32_t>(arg++);
    const uint32_t west_y = get_arg_val<uint32_t>(arg++);
    const uint32_t east_x = get_arg_val<uint32_t>(arg++);
    const uint32_t east_y = get_arg_val<uint32_t>(arg++);

    volatile tt_l1_ptr uint32_t* done_self = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_self);

    constexpr auto grid_args = TensorAccessorArgs<1>();
    const auto grid_a = TensorAccessor(grid_args, grid_a_addr);
    const auto grid_b = TensorAccessor(grid_args, grid_b_addr);

    const uint32_t num_batches = nbx * nby;
    for (uint32_t it = 0; it < num_its; ++it) {
        const auto& grid = (it % 2 == 0) ? grid_b : grid_a;
        for (uint32_t k = 0; k < num_batches; ++k) {
            const uint32_t bx = bx0 + k / nby;
            const uint32_t by = by0 + k % nby;
            // First point of the batch within its padded row; 64-byte aligned on Blackhole.
            const uint32_t col_byte = (PAD + by * BATCH) * ELEM_BYTES;

            cb_wait_front(cb_out, 1);
            const uint32_t page = get_read_ptr(cb_out);
            for (uint32_t r = 0; r < BATCH; ++r) {
                // Result row r is padded grid row bx * 32 + r + 1 (row 0 is the top halo).
                noc_async_write(page + r * PAGE_ROW_BYTES, grid.get_noc_addr(bx * BATCH + r + 1, col_byte), PAGE_ROW_BYTES);
            }
            noc_async_writes_flushed();  // the page has left L1 and can be reused
            cb_pop_front(cb_out, 1);
        }

        if (it + 1 < num_its) {
            // Every write of this sweep must be acknowledged before any reader may use it.
            noc_async_write_barrier();
            noc_semaphore_set(done_self, it + 1);
            // This core is the south neighbour of its north neighbour, and so on.
            if (neighbours & HAS_NORTH) {
                noc_semaphore_inc(get_noc_addr(north_x, north_y, sem_from_south), 1);
            }
            if (neighbours & HAS_SOUTH) {
                noc_semaphore_inc(get_noc_addr(south_x, south_y, sem_from_north), 1);
            }
            if (neighbours & HAS_WEST) {
                noc_semaphore_inc(get_noc_addr(west_x, west_y, sem_from_east), 1);
            }
            if (neighbours & HAS_EAST) {
                noc_semaphore_inc(get_noc_addr(east_x, east_y, sem_from_west), 1);
            }
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
