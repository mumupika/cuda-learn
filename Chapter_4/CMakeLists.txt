cmake_minimum_required(VERSION 3.20)

# project name with languages.
project(Chapter4 LANGUAGES C CXX CUDA)

# find cuda.
find_package(CUDAToolkit REQUIRED)

# set basic compilers.
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

# find OpenCV libraries, with headers.
find_package(OpenCV REQUIRED)

# Include directories and libraries
include_directories(../common)
link_directories(../lib)

# Source file to target.
add_executable(simpleVecAdd simpleVecAdd.cpp)
add_executable(cudaSimpleVecAdd simpleVecAdd.cu)
add_executable(julia_set julia_set.cu)
add_executable(test_opencv test_opencv.cpp)

# link target to OPENGL lib.
set_target_properties(julia_set PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(julia_set PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(julia_set SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(julia_set PUBLIC OpenGL ${OpenCV_LIBS})

target_include_directories(test_opencv SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_directories(test_opencv PUBLIC ${OpenCV_LIBS})