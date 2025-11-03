cmake_minimum_required(VERSION 3.26)

project(cuda-learn CUDA C CXX)

# Adapt for default cmake 3.22 on Linux that will cause a failure on clangd since cuda accepts -isystem= but clangd doesn't.
set(CMAKE_NO_SYSTEM_FROM_IMPORTED 1)
# Set the following for avoid using --option-file=xxx.rsp that clangd does not know what it is.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

add_subdirectory(cuda-by-examples)
add_subdirectory(nccl)
add_subdirectory(mpi-example)
add_subdirectory(nvshmem)