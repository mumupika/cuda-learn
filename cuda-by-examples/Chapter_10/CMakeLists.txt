# allow cmake from 3.26.
cmake_minimum_required(VERSION 3.26)

# Chapter 10.
project(Chapter_10 LANGUAGES C CXX CUDA)

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
add_executable(benchmark benchmark.cu)
add_executable(stream stream.cu)
add_executable(multiStream multiStream.cu)
add_executable(multiStream2 multiStream2.cu)

target_compile_features(benchmark PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(stream PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(stream PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(multiStream PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(multiStream PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(multiStream2 PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(multiStream2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_include_directories(histogram_cpu PUBLIC ./include)
# target_link_libraries(histogram_cpu PUBLIC OpenGL::GL GLUT::GLUT)

# target_include_directories(share_bitmap PUBLIC ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(share_bitmap PUBLIC OpenGL ${OpenCV_LIBS})

# target_compile_options(hello_world PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
# target_compile_features(hello_world PRIVATE cxx_std_17 cuda_std_17)
# set_target_properties(hello_world PROPERTIES CUDA_SEPARABLE_COMPILATION ON)