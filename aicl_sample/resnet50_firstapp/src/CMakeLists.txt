# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(resnet50_firstapp)

# Compile options
add_compile_options(-std=c++14)

set(ENV{INSTALL_DIR} "./build")
set(ENV{CPU_ARCH} "x86")
set(IGIE_ROOT "/home/zhenpeng.wang/lib/igie")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -pthread")

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
    $ENV{INSTALL_DIR}/x86_64-linux/include/
    /opt/sw_home/local/cuda/include/
    ${IGIE_ROOT}/include/
    ${IGIE_ROOT}/3rdparty/dlpack/include/
    ${IGIE_ROOT}/3rdparty/dmlc-core/include/
    /usr/include/
    ./
)

if(target STREQUAL "Simulator_Function")
    add_compile_options(-DFUNC_SIM)
endif()

# add host lib path
link_directories(
    ${IGIE_ROOT}/build/
)

add_compile_options(-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>)

add_executable(main
        main.cpp
        aicl_adaptor_iluvatar.cpp)

if(target STREQUAL "Simulator_Function")
    target_link_libraries(main funcsim)
else()
target_link_libraries(main
    cuda
    cudart
    dl
    tvm
    python3.7m
    stdc++)
endif()

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
