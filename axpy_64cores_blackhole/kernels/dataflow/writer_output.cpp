#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile_index = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = 7;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    constexpr auto s_args_dst = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(s_args_dst, dst_addr, tile_bytes);

    Noc noc;
    CircularBuffer cb_out(cb_id_out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_out.wait_front(1);
        noc.async_write(cb_out, dst, tile_bytes, {}, {.page_id = start_tile_index + i});
        noc.async_write_barrier();
        cb_out.pop_front(1);
        // [REMOVED] cb_pop_front(8, 1): CB 8 is never created on the host
    }
}