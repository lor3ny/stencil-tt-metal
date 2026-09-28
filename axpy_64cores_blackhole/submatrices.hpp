#pragma once


#include "utils.hpp"

#include <chrono>


using namespace std;

//! FUCKING OPTIMIZED, NEED TO BE TESTED
inline double extract_5p_memcpy_optimized(
    bfloat16* __restrict in,
    bfloat16* __restrict out_top,
    bfloat16* __restrict out_left,
    bfloat16* __restrict out_right,
    bfloat16* __restrict out_down,
    int rows,
    int cols
) {

    std::chrono::_V2::system_clock::time_point start_total, end_total;
    std::chrono::duration<double, std::milli> elapsed;
    
    start_total = std::chrono::high_resolution_clock::now();

    int r;
    // int lr_count = (cols-1) * sizeof(bfloat16);
    int ud_count = (rows-1) * cols * sizeof(bfloat16);

    // TOP: Copy rows 0 to rows-2 to out_top[cols, 2*cols, ..., (rows-1)*cols]
    if (rows > 1) {
        //std::memcpy(out_top + cols, in, ud_count);
        std::memcpy(out_top + cols, in, ud_count);
    }

    // DOWN: Copy rows 1 to rows-1 to out_down[0, cols, ..., (rows-2)*cols]
    if (rows > 1) {
        //std::memcpy(out_down, in + cols, ud_count);
        std::memcpy(out_down, in + cols, ud_count);
    }

    // LEFT and RIGHT: Copy cols 1 to cols-1 for each row
    for (r = 0; r < rows; r++) {
        // LEFT: Copy cols 1 to cols-1 to out_left[r*cols + 1, ..., r*cols + cols-1]
        // RIGHT: Copy cols 1 to cols-1 to out_right[r*cols, ..., r*cols + cols-2]
        std::memcpy(out_left+(r*cols)+1, in+(r*cols), (cols-1) * sizeof(bfloat16));
        std::memcpy(out_right+(r*cols), in+(r*cols)+1, (cols-1) * sizeof(bfloat16));
    }

    end_total = std::chrono::high_resolution_clock::now();
    elapsed = end_total - start_total;
    return elapsed.count();
}

//! This function is specialized for star 5-point stencil and tries to integrate padding
void extract_5p_memcpy_singleloop(
    bfloat16* __restrict in,
    bfloat16* __restrict out_top,
    bfloat16* __restrict out_left,
    bfloat16* __restrict out_right,
    bfloat16* __restrict out_down,
    int rows, 
    int cols
) {

    for (int r = 0; r < rows; ++r) {
        // TOP: Copy rows 0 to rows-2 to out_top[cols, 2*cols, ..., (rows-1)*cols]
        if (r < rows - 1) {
            std::memcpy(out_top+((r+1)*cols), in+(r*cols), cols * sizeof(bfloat16));
        }

        // LEFT: Copy cols 1 to cols-1 to out_left[r*cols + 1, ..., r*cols + cols-1]
        std::memcpy(out_left+(r*cols)+1, in+(r*cols), (cols-1) * sizeof(bfloat16));

        // RIGHT: Copy cols 1 to cols-1 to out_right[r*cols, ..., r*cols + cols-2]
        std::memcpy(out_right+(r*cols), in+(r*cols)+1, (cols-1) * sizeof(bfloat16));

        // DOWN: Copy rows 1 to rows-1 to out_down[0, cols, ..., (rows-2)*cols]
        if (r > 0) {
            std::memcpy(out_down+((r-1)*cols), in+(r*cols), cols * sizeof(bfloat16));
        }
    }
}

void extract_5p_memcpy(
    bfloat16* __restrict in,
    bfloat16* __restrict out_top,
    bfloat16* __restrict out_left,
    bfloat16* __restrict out_right,
    bfloat16* __restrict out_down,
    int rows, 
    int cols
) {

    // --------- TOP: remove last row
    for (int r = 0; r < rows-1; r++) {
        std::memcpy(out_top+((r+1)*cols), in+(r*cols), cols * sizeof(bfloat16));
    }

    // --------- LEFT: remove first col
    for (int r = 0; r < rows; r++) {
        std::memcpy(out_left+(r*cols)+1, in+(r*cols), (cols-1) * sizeof(bfloat16));
    }

        // --------- RIGHT: remove first col
    for (int r = 0; r < rows; r++) {
        std::memcpy(out_right+(r*cols), in+(r*cols)+1, (cols-1) * sizeof(bfloat16));
    }

    // --------- DOWN: remove first row
    for (int r = 1; r < rows; r++) {
        std::memcpy(out_down+((r-1)*cols), in+(r*cols), cols * sizeof(bfloat16));
    }
}

void extract_5p_loop(
    bfloat16* __restrict in,
    bfloat16* __restrict up,
    bfloat16* __restrict left,
    bfloat16* __restrict right, 
    bfloat16* __restrict down,  
    int rows, 
    int cols
) {
    
    // int flat, i, j;

    for (int i = 0; i < rows; i++) {
        bfloat16* out_up    = &up[(i+1) * cols];
        bfloat16* out_left  = &left[i * cols];
        bfloat16* out_right = &right[i * cols];
        bfloat16* out_down  = &down[(i-1) * cols];

        bfloat16* in_center = &in[i*cols];  // aligned with (j+1)

        for (int j = 0; j < cols; j++) {
            if(i != rows-1){
                out_up[j]    = in_center[j];  // UP
            }
            if(i != 0){
                out_down[j]  = in_center[j];  // DOWN
            }
            if(j != cols-1){
                out_left[j+1]  = in_center[j];  // LEFT
            }
            if(j != 0){
                out_right[j-1] = in_center[j];  // RIGHT
            }
        }
    }
}

//! This function is specialized for star 5-point stencil
// in is 10x10 and out is 64x32,
void extract_submats_5p(
    bfloat16* __restrict in,
    bfloat16* __restrict up,
    bfloat16* __restrict left,
    bfloat16* __restrict right, 
    bfloat16* __restrict down,  
    int rows, 
    int cols, 
    int cols_pad) {
    
    // int flat, i, j;

    for (int i = 0; i < rows; i++) {
        bfloat16* out_up    = &up[i * cols];
        bfloat16* out_left  = &left[i * cols];
        bfloat16* out_right = &right[i * cols];
        bfloat16* out_down  = &down[i * cols];

        bfloat16* in_center = &in[(i + 1) * cols_pad + 1];  // aligned with (j+1)

        for (int j = 0; j < cols; j++) {
            out_up[j]    = in_center[j - cols_pad];  // UP
            out_left[j]  = in_center[j - 1];         // LEFT
            out_right[j] = in_center[j + 1];         // RIGHT
            out_down[j]  = in_center[j + cols_pad];  // DOWN
        }
    }
}

// it is implemented considering star stencils, not squared ones
void pad_with_zeros(const bfloat16* __restrict in, bfloat16* __restrict out, int rows, int cols, int pad_size){

    // size_t new_rows = rows + pad_size*2;
    size_t new_cols = cols + pad_size*2;


    for (size_t r = 0; r < rows; ++r) {
        const bfloat16* src = in + r * cols;
        bfloat16* dest = out + (r + pad_size) * new_cols + pad_size;

        std::memcpy(dest, src, cols * sizeof(bfloat16));
    }
}


uint32_t align_vector_size(vector<bfloat16>& in, size_t starting_size, size_t single_tile_size){

    uint32_t new_size = starting_size;

    if(new_size % single_tile_size != 0){
        uint32_t align = 1;
        while ((new_size + align) % single_tile_size != 0){
            align += 1;
        }    
        new_size += align;
    }
    uint32_t new_count = new_size / sizeof(bfloat16);
    in.resize(new_count, 0.0f);

    return new_size;
}


//! Tester function
void matVecMul(
    const std::vector<float>& matrix,
    const float* vec,
    float* result,
    size_t rows,
    size_t cols
) {
    for (size_t i = 0; i < rows; ++i) {
        result[i] = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            result[i] += matrix[i * cols + j] * vec[j];
        }
    }
}

