cmake_minimum_required(VERSION 3.26)

project(nccl-examples C CXX CUDA)

# Find openmpi first.
find_package(MPI)

message("compute arch: ${CMAKE_CUDA_ARCHITECTURES}")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR NOT ${CMAKE_CUDA_ARCHITECTURES} STREQUAL "89")
    set(CMAKE_CUDA_ARCHITECTURES "89")
endif()

# Display them if we found something...
if (MPI_FOUND)
    message(STATUS "MPI FOUND!")
    message(STATUS "MPI HOME is: ${MPI_HOME}")
    message(STATUS "MPI INCLUDE C DIRECTORY IS: ${MPI_C_INCLUDE_DIRS}")
    message(STATUS "MPI INCLUDE CXX DIRECTORY IS: ${MPI_CXX_INCLUDE_DIRS}")
    message(STATUS "MPI C LIBS:${MPI_C_LIBRARIES}")
    message(STATUS "MPI C LIBS:${MPI_CXX_LIBRARIES}")
else ()
    message(WARNING "NOT FOUND...")
endif ()

# find cuda toolkit.
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDA FOUND!")
    message(STATUS "Available cuda compiler: ${CUDAToolkit_NVCC_EXECUTABLE}")
else ()
    message(STATUS "Failed to find cuda!")
endif ()

# find NCCL.
message(STATUS "NCCL not found in standard place.")
set(NCCL_INCLUDE_DIR "/usr/include")
set(NCCL_LIB "/usr/lib/x86_64-linux-gnu/libnccl.so")

file(GLOB SOURCES "src/*.cu" "src/*.cpp" "src/*.c" "src/*.cxx")
foreach (source ${SOURCES})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    if (MPI_FOUND)
        target_include_directories(${name} PUBLIC  ${MPI_CXX_INCLUDE_DIRS})
        target_link_libraries(${name} PUBLIC ${MPI_CXX_LIBRARIES} )
    endif ()
    if (${name} STREQUAL "fault_tolerance") 
        target_compile_features(${name} PUBLIC cxx_std_17 cuda_std_17)
    endif()
    target_include_directories(${name} PUBLIC ${NCCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS} "./include")
    target_link_libraries(${name} PUBLIC ${NCCL_LIB} CUDA::cuda_driver CUDA::cudart)
endforeach()