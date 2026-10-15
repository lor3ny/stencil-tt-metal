cmake_minimum_required(VERSION 3.22...4.2)
project(optimized_axpy_stencil LANGUAGES CXX)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# The CPU baselines are timed, so build optimised unless asked otherwise.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
find_package(Threads REQUIRED)

add_executable(optimized_axpy_stencil main.cpp)
target_compile_features(optimized_axpy_stencil PRIVATE cxx_std_20)
target_compile_options(optimized_axpy_stencil PRIVATE -march=native)  # lets golden_optimized use the host's widest SIMD
target_link_libraries(optimized_axpy_stencil PRIVATE TT::Metalium Threads::Threads)
# tt-metal compiles the kernels when the program first runs; give it their absolute location so the
# binary can be started from any directory (OPTIMIZED_AXPY_STENCIL_KERNEL_DIR in the environment overrides it).
target_compile_definitions(optimized_axpy_stencil PRIVATE OPTIMIZED_AXPY_STENCIL_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")

