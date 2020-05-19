cmake_minimum_required(VERSION 3.10)

project(ExaMiniMD LANGUAGES CXX)

find_package(Kokkos 3.0 REQUIRED)
option(USE_MPI "MPI" ON)

if (USE_MPI)
  find_package(MPI REQUIRED CXX)
else()
  string(FIND "${Kokkos_DEVICES}" "Cuda" USING_CUDA)
  # string FIND returns -1 if not found
  if (NOT USING_CUDA STREQUAL "-1")
    if (CMAKE_CXX_COMPILER_ID MATCHES "nvcc")
      enable_language(CUDA)
      set(USE_CUDA True INTERNAL "")
    else()
      message(FATAL_ERROR "Kokkos uses CUDA but compiler is not nvcc")
    endif()
  endif()
endif()

# force target_source to resolve relative paths immediately
if(POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

add_subdirectory(src)
add_subdirectory(input)
