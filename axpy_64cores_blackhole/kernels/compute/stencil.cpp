#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {  // [CHANGED] was: namespace NAMESPACE { void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_inUP = 1, cb_inLEFT = 2, cb_inRIGHT = 3, cb_inDOWN = 4, cb_SCALAR = 5;
    constexpr uint32_t cb_MID = 6;     // AUXILIARY
    constexpr uint32_t cb_OUTPUT = 7;  // OUTPUT
    constexpr uint32_t dst0 = 0;

    compute_kernel_hw_startup(cb_inUP, cb_inDOWN, cb_MID);  // [CHANGED] was binary_op_init_common

    for (uint32_t i = 0; i < num_tiles; i++) {

        // UP + DOWN -> MID
        tile_regs_acquire();
        add_init(cb_inUP, cb_inDOWN);  // [CHANGED] was add_tiles_init
        cb_wait_front(cb_inUP, 1);
        cb_wait_front(cb_inDOWN, 1);
        add_tiles(cb_inUP, cb_inDOWN, 0, 0, dst0);
        cb_pop_front(cb_inUP, 1);
        cb_pop_front(cb_inDOWN, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_MID, 1);
        pack_tile(dst0, cb_MID);
        cb_push_back(cb_MID, 1);
        tile_regs_release();

        // LEFT + MID -> MID
        tile_regs_acquire();
        add_init(cb_inLEFT, cb_MID);
        cb_wait_front(cb_inLEFT, 1);
        cb_wait_front(cb_MID, 1);
        add_tiles(cb_inLEFT, cb_MID, 0, 0, dst0);
        cb_pop_front(cb_inLEFT, 1);
        cb_pop_front(cb_MID, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_MID, 1);
        pack_tile(dst0, cb_MID);
        cb_push_back(cb_MID, 1);
        tile_regs_release();

        // RIGHT + MID -> MID
        tile_regs_acquire();
        add_init(cb_inRIGHT, cb_MID);
        cb_wait_front(cb_inRIGHT, 1);
        cb_wait_front(cb_MID, 1);
        add_tiles(cb_inRIGHT, cb_MID, 0, 0, dst0);
        cb_pop_front(cb_inRIGHT, 1);
        cb_pop_front(cb_MID, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_MID, 1);
        pack_tile(dst0, cb_MID);
        cb_push_back(cb_MID, 1);
        tile_regs_release();

        // SCALAR * MID -> OUTPUT
        tile_regs_acquire();
        mul_init(cb_SCALAR, cb_MID);  // [CHANGED] was mul_tiles_init
        cb_wait_front(cb_SCALAR, 1);  // never popped: reused every iteration
        cb_wait_front(cb_MID, 1);
        mul_tiles(cb_SCALAR, cb_MID, 0, 0, dst0);
        cb_pop_front(cb_MID, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_OUTPUT, 1);
        pack_tile(dst0, cb_OUTPUT);
        cb_push_back(cb_OUTPUT, 1);
        tile_regs_release();
    }
}