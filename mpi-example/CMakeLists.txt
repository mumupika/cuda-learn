cmake_minimum_required(VERSION 3.26)

project(nccl-examples C CXX CUDA)

# Find openmpi first.
find_package(MPI)

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

include_directories("./include")

file(GLOB SOURCES "src/*.cc" "src/*.cpp" "src/*.c" "src/*.cxx")
foreach (source ${SOURCES})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_include_directories(${name} PUBLIC  ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(${name} PUBLIC ${MPI_CXX_LIBRARIES})
endforeach()