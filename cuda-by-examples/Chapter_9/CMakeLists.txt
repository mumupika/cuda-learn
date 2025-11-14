# allow cmake from 3.26.
cmake_minimum_required(VERSION 3.26)

# Chapter 5.
project(Chapter_5 LANGUAGES C CXX CUDA)

# find cuda.
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenCV REQUIRED)

# set specific compilers.
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# Set the C++ standard.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set cuda arch with cuda compile flags.
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# include directories and libraries.
include_directories(../common ./include)
link_directories(../lib)

# source to target.
add_executable(histogram_cpu histogram_cpu.cu)
add_executable(histogram_gpu histogram_gpu.cu)

target_compile_features(histogram_gpu PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(histogram_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_include_directories(histogram_cpu PUBLIC ./include)
# target_link_libraries(histogram_cpu PUBLIC OpenGL::GL GLUT::GLUT)

# target_include_directories(share_bitmap PUBLIC ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(share_bitmap PUBLIC OpenGL ${OpenCV_LIBS})