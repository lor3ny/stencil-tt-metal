// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute (TRISC 0-2). For every batch: DST = E + W, then DST += S and DST += N with DST reused as
// one operand, then DST *= 0.25 on the SFPU, and the result is packed into c_16. The partial sums
// stay in the DST register (fp32, see fp32_dest_acc_en on the host), so each batch is unpacked and
// packed once.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);  // batches owned by this core x sweeps

    constexpr uint32_t cb_east = tt::CBIndex::c_0;
    constexpr uint32_t cb_west = tt::CBIndex::c_1;
    constexpr uint32_t cb_south = tt::CBIndex::c_2;
    constexpr uint32_t cb_north = tt::CBIndex::c_3;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t dst = 0;
    constexpr uint32_t quarter = 0x3E800000;  // 0.25f as fp32 bits

    compute_kernel_hw_startup(cb_east, cb_west, cb_out);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_wait_front(cb_east, 1);
        cb_wait_front(cb_west, 1);
        cb_wait_front(cb_south, 1);
        cb_wait_front(cb_north, 1);

        tile_regs_acquire();
        add_init(cb_east, cb_west);
        add_tiles(cb_east, cb_west, 0, 0, dst);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_south);
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_south, 0, dst);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_north);
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_north, 0, dst);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, quarter);
        tile_regs_commit();

        cb_pop_front(cb_east, 1);
        cb_pop_front(cb_west, 1);
        cb_pop_front(cb_south, 1);
        cb_pop_front(cb_north, 1);

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(dst, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
