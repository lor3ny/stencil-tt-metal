cmake_minimum_required(VERSION 3.22...4.2)
project(axpy_stencil)

add_executable(axpy_stencil)
target_sources(axpy_stencil PRIVATE tt_main.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(axpy_stencil PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(axpy_stencil TT::CommonPCH)
endif()
