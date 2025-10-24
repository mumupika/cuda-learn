cmake_minimum_required(VERSION 3.20)

project(cuda-learn CUDA C CXX)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

add_subdirectory(./Chapter_3)
add_subdirectory(./Chapter_4)
add_subdirectory(./Chapter_5)
add_subdirectory(./nccl)