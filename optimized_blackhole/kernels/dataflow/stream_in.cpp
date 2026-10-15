// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader (RISCV_1, NoC 1). For every sweep and every 32 x 32 batch this core owns, it reads the
// batch's 34 x 34 window (the batch plus a one-point halo) from DRAM into L1 scratch and copies it
// into four 32 x 32 pages holding each point's east, west, south and north neighbour. Windows are
// double buffered: batch k + 1 is read from DRAM while batch k is being copied.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t BATCH = 32;                           // batch edge in grid points
constexpr uint32_t WINDOW = BATCH + 2;                   // window edge, including the halo
constexpr uint32_t ELEM_BYTES = 2;                       // bf16
constexpr uint32_t PAGE_ROW_BYTES = BATCH * ELEM_BYTES;  // one row of a 32 x 32 page

// DRAM read alignment from the host HAL: 64 B on Blackhole. A read starts on this boundary and lands
// at the same offset within it in L1. The host pads every grid row with ALIGN / 2 columns per side,
// so interior rows start on the boundary too.
constexpr uint32_t ALIGN = get_compile_time_arg_val(0);
constexpr uint32_t PAD = ALIGN / ELEM_BYTES;
static_assert(ALIGN >= 16 && (ALIGN & (ALIGN - 1)) == 0, "DRAM alignment must be a power of two of at least 16");

// One window row in L1: up to ALIGN - 2 leading bytes plus 34 points, rounded up so that the next
// row starts aligned again (192 B on Blackhole).
constexpr uint32_t LINE_BYTES = (ALIGN + WINDOW * ELEM_BYTES + ALIGN - 1) / ALIGN * ALIGN;
constexpr uint32_t WINDOW_BYTES = WINDOW * LINE_BYTES;

constexpr uint32_t cb_east = tt::CBIndex::c_0;    // u[i][j+1]
constexpr uint32_t cb_west = tt::CBIndex::c_1;    // u[i][j-1]
constexpr uint32_t cb_south = tt::CBIndex::c_2;   // u[i+1][j]
constexpr uint32_t cb_north = tt::CBIndex::c_3;   // u[i-1][j]
constexpr uint32_t cb_window = tt::CBIndex::c_4;  // L1 scratch for two windows

constexpr uint32_t HAS_NORTH = 1, HAS_SOUTH = 2, HAS_WEST = 4, HAS_EAST = 8;

// Byte offset, within a padded grid row, of the west halo point of batch column `by`.
FORCE_INLINE uint32_t halo_byte(uint32_t by) { return (PAD + by * BATCH - 1) * ELEM_BYTES; }

// Issue the 34 row reads of the window around batch (bx, by) into L1 at `l1_dst`.
template <typename Grid>
FORCE_INLINE void read_window(const Grid& grid, uint32_t bx, uint32_t by, uint32_t l1_dst) {
    const uint32_t start = halo_byte(by) & ~(ALIGN - 1);
    const uint32_t bytes = halo_byte(by) - start + WINDOW * ELEM_BYTES;
    for (uint32_t i = 0; i < WINDOW; ++i) {
        // Window row i is padded grid row bx * 32 + i: the halo row above the batch, the batch's
        // 32 rows, then the halo row below it.
        noc_async_read(grid.get_noc_addr(bx * BATCH + i, start), l1_dst + i * LINE_BYTES, bytes);
    }
}

// Copy one 64-byte page row. `dst_addr` is word aligned. `src_addr` is word aligned or two bytes past
// a word boundary; in that case each output word is spliced from two neighbouring input words.
FORCE_INLINE void copy_row(uint32_t dst_addr, uint32_t src_addr) {
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    if ((src_addr & 3) == 0) {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
        for (uint32_t w = 0; w < PAGE_ROW_BYTES / 4; ++w) {
            dst[w] = src[w];
        }
    } else {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr - 2);
        uint32_t lo = src[0];
        for (uint32_t w = 0; w < PAGE_ROW_BYTES / 4; ++w) {
            const uint32_t hi = src[w + 1];
            dst[w] = (lo >> 16) | (hi << 16);
            lo = hi;
        }
    }
}

}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t grid_a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grid_b_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t bx0 = get_arg_val<uint32_t>(arg++);  // first batch row owned by this core
    const uint32_t nbx = get_arg_val<uint32_t>(arg++);  // number of batch rows
    const uint32_t by0 = get_arg_val<uint32_t>(arg++);  // first batch column
    const uint32_t nby = get_arg_val<uint32_t>(arg++);  // number of batch columns
    const uint32_t num_its = get_arg_val<uint32_t>(arg++);
    // Each semaphore counts finished sweeps: of this core's writer, and of each neighbour's writer.
    const uint32_t sem_self = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t sem_north = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t sem_south = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t sem_west = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t sem_east = get_semaphore(get_arg_val<uint32_t>(arg++));
    const uint32_t neighbours = get_arg_val<uint32_t>(arg++);

    volatile tt_l1_ptr uint32_t* done_self = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_self);
    volatile tt_l1_ptr uint32_t* done_north = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_north);
    volatile tt_l1_ptr uint32_t* done_south = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_south);
    volatile tt_l1_ptr uint32_t* done_west = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_west);
    volatile tt_l1_ptr uint32_t* done_east = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_east);

    // Both grids have the same layout: one DRAM page per padded row, interleaved over all banks.
    constexpr auto grid_args = TensorAccessorArgs<1>();
    const auto grid_a = TensorAccessor(grid_args, grid_a_addr);
    const auto grid_b = TensorAccessor(grid_args, grid_b_addr);

    // The scratch buffer has a single page, held for the whole kernel. Its base is rounded up to the
    // DRAM alignment (the host sized the page with that much slack).
    cb_reserve_back(cb_window, 1);
    const uint32_t scratch = (get_write_ptr(cb_window) + ALIGN - 1) & ~(ALIGN - 1);

    const uint32_t num_batches = nbx * nby;
    for (uint32_t it = 0; it < num_its; ++it) {
        if (it > 0) {
            // Sweep `it` reads what sweep it - 1 wrote: this core's own rows, and the halo rows and
            // columns of its neighbours. Wait until all of those writers have finished that sweep.
            noc_semaphore_wait_min(done_self, it);
            if (neighbours & HAS_NORTH) {
                noc_semaphore_wait_min(done_north, it);
            }
            if (neighbours & HAS_SOUTH) {
                noc_semaphore_wait_min(done_south, it);
            }
            if (neighbours & HAS_WEST) {
                noc_semaphore_wait_min(done_west, it);
            }
            if (neighbours & HAS_EAST) {
                noc_semaphore_wait_min(done_east, it);
            }
        }
        // Even sweeps read A and write B; odd sweeps do the reverse.
        const auto& grid = (it % 2 == 0) ? grid_a : grid_b;

        read_window(grid, bx0, by0, scratch);
        for (uint32_t k = 0; k < num_batches; ++k) {
            const uint32_t by = by0 + k % nby;
            const uint32_t window = scratch + (k % 2) * WINDOW_BYTES;

            noc_async_read_barrier();  // window k has landed
            if (k + 1 < num_batches) {
                // Read window k + 1 into the other half while window k is copied.
                read_window(grid, bx0 + (k + 1) / nby, by0 + (k + 1) % nby, scratch + ((k + 1) % 2) * WINDOW_BYTES);
            }

            cb_reserve_back(cb_east, 1);
            cb_reserve_back(cb_west, 1);
            cb_reserve_back(cb_south, 1);
            cb_reserve_back(cb_north, 1);
            const uint32_t east = get_write_ptr(cb_east);
            const uint32_t west = get_write_ptr(cb_west);
            const uint32_t south = get_write_ptr(cb_south);
            const uint32_t north = get_write_ptr(cb_north);

            // L1 address of the west halo point in window row 0. Window row r + 1 holds grid row r of
            // the batch: point j of that row sits at halo + (r + 1) * LINE_BYTES + (j + 1) * 2.
            const uint32_t halo = window + (halo_byte(by) & (ALIGN - 1));
            for (uint32_t r = 0; r < BATCH; ++r) {
                const uint32_t row = r * PAGE_ROW_BYTES;
                copy_row(east + row, halo + (r + 1) * LINE_BYTES + 2 * ELEM_BYTES);
                copy_row(west + row, halo + (r + 1) * LINE_BYTES);
                copy_row(south + row, halo + (r + 2) * LINE_BYTES + ELEM_BYTES);
                copy_row(north + row, halo + r * LINE_BYTES + ELEM_BYTES);
            }

            cb_push_back(cb_east, 1);
            cb_push_back(cb_west, 1);
            cb_push_back(cb_south, 1);
            cb_push_back(cb_north, 1);
        }
    }

    // Hand the scratch page back so the circular buffer ends the program empty.
    cb_push_back(cb_window, 1);
    cb_wait_front(cb_window, 1);
    cb_pop_front(cb_window, 1);
}
