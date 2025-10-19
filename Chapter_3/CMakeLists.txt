cmake_minimum_required(VERSION 3.20)

project(simplePrintf LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../common)
link_directories(../lib)

# Source file
# Add target for simplePrintf
add_executable(hello_world hello_world.cu)
add_executable(kernel_call kernel_call.cu)
add_executable(pass_param pass_param.cu)
add_executable(query_device query_device.cu)

# target_compile_options(hello_world PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# target_compile_features(hello_world PRIVATE cxx_std_17 cuda_std_17)

# set_target_properties(hello_world PROPERTIES CUDA_SEPARABLE_COMPILATION ON)