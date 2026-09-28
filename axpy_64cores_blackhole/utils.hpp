#pragma once

#include <iostream>
#include <tt-metalium/bfloat16.hpp>
#include <iostream>
#include <vector>



/*--------------------------------

    GENERIC UTILITIES

----------------------------------*/




inline void printMat(std::vector<bfloat16>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << static_cast<float>(matrix[i * cols + j]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
}


inline void printMat(const bfloat16* matrix, int rows, int cols) {
    for(int i =0; i<rows; i++){
        for(int j =0; j<cols; j++){
            std::cout << static_cast<float>(matrix[i*cols + j]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
}


inline void InitializeGrid(double *grid, int dim){
    int i, j;
    for(i = 0; i < dim; ++i){
        for(j = 0; j < dim; ++j){
            grid[i*dim +j] = 0.0;
        }
    }
}

inline void PrintGrid(double *grid, int dim){
    int i, j;
    for(i = 0; i < dim; ++i){
        for(j = 0; j < dim; ++j){
            std::cout << " " << grid[i*dim + j] << " ";
        }
        std::cout << std::endl;
    }
}


void golden_stencil(std::vector<bfloat16>& input, std::vector<bfloat16>& output, int rows, int cols, int num_its) {
    
    std::vector<bfloat16> in_copy = input;
    
    bfloat16* in_ptr = in_copy.data();
    bfloat16* out_ptr = output.data();

    std::chrono::_V2::system_clock::time_point start_total, end_total;
    std::chrono::duration<double, std::milli> elapsed;
    
    start_total = std::chrono::high_resolution_clock::now();
    for (int k=0;k<num_its;k++) {
        for (int i=1; i<rows-1; i++) {
            for (int j=1; j<cols-1; j++) {
                out_ptr[(i*cols)+j] = bfloat16((float) static_cast<float>(0.25*(in_ptr[((i-1)*cols)+j]) 
                                        + static_cast<float>(in_ptr[((i+1)*cols)+j]) 
                                        + static_cast<float>(in_ptr[(i*cols)+(j-1)]) 
                                        + static_cast<float>(in_ptr[(i*cols)+(j+1)])));
            }
        }
        out_ptr[(rows/2)*cols + cols/2] = bfloat16(100.0f);
        bfloat16* tmp = out_ptr;
        out_ptr=in_ptr;
        in_ptr=tmp;
    }
    end_total = std::chrono::high_resolution_clock::now();
    elapsed = end_total - start_total;

    // if(num_its % 2 == 0) {
    //     printMat(in_copy, rows, cols);
    // } else {
    //     printMat(output, rows, cols);
    // }

    std::cout << "-CPU_BASELINE- " << elapsed.count() << " ms" << std::endl;
}

//! Not it isn't used, will be checked if needed
// inline void saveGridCSV(const std::string& filename, double* grid, int dim_x, int dim_y) {
//     std::ofstream file(filename);
//     if (!file) {
//         std::cerr << "Errore nell'apertura del file" << std::endl;
//         return;
//     }
//     file << dim_x << " " << dim_y << endl;
//     for (size_t i=0; i<dim_x; ++i) {
//         for (size_t j = 0; j < dim_y; ++j) {
//             file << grid[i*dim_x + j] << (j + 1 == dim_y ? "\n" : " ");
//         }
//     }
//     file.close();
// }

/*--------------------------------

    TENSTORRENT UTILITIES

----------------------------------*/


// IMPORTANT
// Tenstorrent works with BFP16, TT-Metalium requires uint32_t buffers, and packs on it two BFP16.
// So it packs on every uint32_t cel two bfloat16 values. Checks printf("Result = %x\n", result_vec[0]);
// You will see the two copies of the same HEX. 

// void tt_printMatrix(uint32_t* matrix, size_t rows, size_t cols) {
//     bfloat16* a_bf16 = reinterpret_cast<bfloat16*>(matrix);

//     for(int i =0; i<rows; i++){
//         for(int j =0; j<cols; j++){
//             std::cout << static_cast<float>(a_bf16[i*cols + j]) << " ";
//         }
//         std::cout << "\n";
//     }
//     std::cout << std::flush;
// }