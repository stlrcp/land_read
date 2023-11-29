# Copyright (c) Iluvatar Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(aicl_execute_add)

# Compile options
add_compile_options(-std=c++14)

set(ENV{INSTALL_DIR} "./build")
set(ENV{CPU_ARCH} "x86")
set(IGIE_ROOT "/home/zhenpeng.wang/lib/igie")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../run/out")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../outputs")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

if (NOT DEFINED ENV{INSTALL_DIR})
    message(FATAL_ERROR "Not Defined INSTALL_DIR")
endif()

if (NOT DEFINED ENV{CPU_ARCH})
    message(FATAL_ERROR "Not Defined CPU_ARCH")
endif()

if($ENV{CPU_ARCH} MATCHES "aarch64")
    message("arm architecture detected")
    if(EXISTS "$ENV{INSTALL_DIR}/driver/libmedia_mini.so")
        message("target aarch64 200dk")
    else()
        message("target aarch64 300")
    endif()
else()
    message("x86 architecture detected")
    message("target x86 300")
endif()

# Header path
include_directories(
    /opt/sw_home/local/cuda/include/
    ${IGIE_ROOT}/include/
    ${IGIE_ROOT}/3rdparty/dlpack/include/
    ${IGIE_ROOT}/3rdparty/dmlc-core/include/
    /usr/include/
    ../inc
)

# add host lib path
link_directories(
    ${IGIE_ROOT}/build/
)

add_executable(execute_add_op
        operator_desc.cpp
        op_runner.cpp
        main.cpp
        aicl_adaptor_iluvatar.cpp
        common.cpp)

target_link_libraries(execute_add_op
        cuda
        cudart
        dl
        tvm
        python3.7m
        stdc++)

install(TARGETS execute_add_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
