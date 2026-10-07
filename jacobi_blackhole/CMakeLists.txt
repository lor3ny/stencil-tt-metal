cmake_minimum_required(VERSION 3.22...4.2)
project(jacobi_blackhole LANGUAGES CXX)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

add_executable(jacobi jacobi.cpp)
target_compile_features(jacobi PRIVATE cxx_std_20)
target_link_libraries(jacobi PRIVATE TT::Metalium)
# tt-metal compiles the kernels when the program first runs; give it their absolute location so the
# binary can be started from any directory (JACOBI_KERNEL_DIR in the environment overrides it).
target_compile_definitions(jacobi PRIVATE JACOBI_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
