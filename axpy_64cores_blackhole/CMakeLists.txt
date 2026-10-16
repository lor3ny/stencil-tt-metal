cmake_minimum_required(VERSION 3.22...4.2)
project(axpy_stencil)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# The CPU baselines are timed, so build optimised unless asked otherwise.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
find_package(Threads REQUIRED)

add_executable(axpy_stencil)
target_sources(axpy_stencil PRIVATE tt_main.cpp)
target_compile_features(axpy_stencil PRIVATE cxx_std_20)  # std::barrier in golden_optimized
target_compile_options(axpy_stencil PRIVATE -march=native)  # lets golden_optimized use the host's widest SIMD

target_link_libraries(axpy_stencil PUBLIC TT::Metalium Threads::Threads)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(axpy_stencil TT::CommonPCH)
endif()
