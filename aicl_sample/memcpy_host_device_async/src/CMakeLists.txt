# Copyright © 2023 Iluvatar CoreX. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
message(STATUS "Current cmake version: ${CMAKE_MAJOR_VERSION}."
                "${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}")

# project information
project(async_memcpy_host2device)

# find CUDA
find_package(CUDA)

# set CUDA_PATH
if(DEFINED CUDA_PATH)
    set(CUDA_PATH "${CUDA_PATH}")
else()
    set(CUDA_PATH "/opt/sw_home/local/cuda/")
endif()
message(STATUS "CUDA_PATH=${CUDA_PATH}")


# set COREX_ARCH
if (DEFINED COREX_ARCH)
    set(CMAKE_CUDA_ARCH "${COREX_ARCH}")
else()
    set(CMAKE_CUDA_ARCH ivcore10)
endif()
message(STATUS "CUDA_ARCH=${CMAKE_CUDA_ARCH}")

# clang++ Compile
set(CMAKE_CXX_COMPILER "${CUDA_PATH}/../bin/clang++")
set(CMAKE_C_COMPILER "${CUDA_PATH}/../bin/clang")
set(CUDA_COMPILER "${CUDA_PATH}/../bin/clang++")

# Compile options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS " --cuda-gpu-arch=${CMAKE_CUDA_ARCH} -lcuda -lcudart")

message(STATUS "CUDA compiler ID : ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

enable_language(CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./")
set(RUN_NAME "main")


# Header path
include_directories("${CUDA_PATH}/include/")


add_executable("${RUN_NAME}" async_memcpy_host2device.cpp)
target_include_directories("${RUN_NAME}" PRIVATE "${CUDA_PATH}/include/")


# add lib path
add_library(iluvatar STATIC aicl_adaptor_iluvatar.cpp)

# link lib
target_link_libraries("${RUN_NAME}" iluvatar)

install(TARGETS "${RUN_NAME}" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

