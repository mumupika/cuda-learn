# allow cmake from 3.26.
cmake_minimum_required(VERSION 3.26)

# Chapter 6.
project(Chapter_6 LANGUAGES C CXX CUDA)

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
add_executable(ray_tracing ray_tracing.cu)
target_include_directories(ray_tracing PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ray_tracing PUBLIC OpenGL ${OpenCV_LIBS})

add_executable(ray_tracing_const ray_tracing_const.cu)
target_include_directories(ray_tracing_const PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ray_tracing_const PUBLIC OpenGL ${OpenCV_LIBS})
# target_include_directories(ray_tracing PUBLIC ./include)
# target_link_libraries(ray_tracing PUBLIC OpenGL::GL GLUT::GLUT)