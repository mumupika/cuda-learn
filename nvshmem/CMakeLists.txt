cmake_minimum_required(VERSION 3.26)

project(nvshmem_examples LANGUAGES C CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR NOT ${CMAKE_CUDA_ARCHITECTURES} STREQUAL "89")
    set(CMAKE_CUDA_ARCHITECTURES "89")
endif()

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(NVSHMEM REQUIRED)

include_directories(${NVSHMEM_INCLUDE_DIR} ${MPI_CXX_INCLUDE_DIR})
link_directories(${NVSHMEM_INCLUDE_DIR} ${MPI_CXX_LIBRARIES})


add_executable(basic_init basic_init.cpp)
# target_compile_options(basic_init PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
# target_compile_features(basic_init PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(basic_init PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(basic_init 
    PUBLIC
    CUDA::cudart 
    CUDA::cuda_driver 
    nvshmem::nvshmem_host
    nvshmem::nvshmem_device
)

