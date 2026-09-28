#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"


void kernel_main() {

    uint32_t src_addr_UP = get_arg_val<uint32_t>(0);
    uint32_t src_addr_LEFT = get_arg_val<uint32_t>(1);
    uint32_t src_addr_RIGHT = get_arg_val<uint32_t>(2);
    uint32_t src_addr_DOWN = get_arg_val<uint32_t>(3);
    uint32_t src_addr_SCALAR = get_arg_val<uint32_t>(4);
    uint32_t start_tile_index = get_arg_val<uint32_t>(5);
    uint32_t scalar_tile_index = get_arg_val<uint32_t>(6);
    uint32_t num_tiles = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_1 = 1; // UP
    constexpr uint32_t cb_id_2 = 2; // LEFT
    constexpr uint32_t cb_id_3 = 3; // RIGHT
    constexpr uint32_t cb_id_4 = 4; // DOWN
    constexpr uint32_t cb_id_5 = 5; // SCALAR

    const uint32_t tile_bytes = get_tile_size(cb_id_1);  // all CBs hold bf16 32x32 tiles

    // [CHANGED] same order the host appends them: scalar, up, left, right, down
    constexpr auto s_args_SCALAR = TensorAccessorArgs<0>();
    constexpr auto s_args_UP     = TensorAccessorArgs<s_args_SCALAR.next_compile_time_args_offset()>();
    constexpr auto s_args_LEFT   = TensorAccessorArgs<s_args_UP.next_compile_time_args_offset()>();
    constexpr auto s_args_RIGHT  = TensorAccessorArgs<s_args_LEFT.next_compile_time_args_offset()>();
    constexpr auto s_args_DOWN   = TensorAccessorArgs<s_args_RIGHT.next_compile_time_args_offset()>();

    const auto src_SCALAR = TensorAccessor(s_args_SCALAR, src_addr_SCALAR, tile_bytes);
    const auto src_UP     = TensorAccessor(s_args_UP, src_addr_UP, tile_bytes);
    const auto src_LEFT   = TensorAccessor(s_args_LEFT, src_addr_LEFT, tile_bytes);
    const auto src_RIGHT  = TensorAccessor(s_args_RIGHT, src_addr_RIGHT, tile_bytes);
    const auto src_DOWN   = TensorAccessor(s_args_DOWN, src_addr_DOWN, tile_bytes);

    Noc noc;
    CircularBuffer cb_up(cb_id_1), cb_left(cb_id_2), cb_right(cb_id_3), cb_down(cb_id_4), cb_scalar(cb_id_5);

    // Scalar tile: read once; compute waits on it every iteration but never pops it
    cb_scalar.reserve_back(1);
    noc.async_read(src_SCALAR, cb_scalar, tile_bytes, {.page_id = scalar_tile_index}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_scalar.push_back(1);

    for (uint32_t i = 0; i < num_tiles; i++) {
        const uint32_t tile_id = start_tile_index + i;

        cb_up.reserve_back(1);
        noc.async_read(src_UP, cb_up, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_up.push_back(1);

        cb_down.reserve_back(1);
        noc.async_read(src_DOWN, cb_down, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_down.push_back(1);

        cb_left.reserve_back(1);
        noc.async_read(src_LEFT, cb_left, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_left.push_back(1);

        cb_right.reserve_back(1);
        noc.async_read(src_RIGHT, cb_right, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_right.push_back(1);
    }
}