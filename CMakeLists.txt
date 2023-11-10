cmake_minimum_required(VERSION 3.10)

project(ExaMiniMD LANGUAGES CXX)

include(GNUInstallDirs)

find_package(Kokkos 3.0 REQUIRED)

if ((Kokkos_DIR) AND (NOT Kokkos_ROOT))
  message(STATUS "STATUS using lib64/cmake/Kokkos/KokkosConfig.cmake file (in Kokkos installation point) and Kokkos_DIR set to: ${Kokkos_DIR}")
else()
  message(STATUS "STATUS using Kokkos installation point with Kokkos_ROOT: ${Kokkos_ROOT}")
  message(STATUS "STATUS Kokkos_DIR: ${Kokkos_DIR} automatically set when using Kokkos_ROOT")
endif()

option(USE_MPI "Build with MPI" ON)

if (USE_MPI)
  find_package(MPI REQUIRED CXX)
else()
  kokkos_check(DEVICES Cuda RETURN_VALUE USE_CUDA)
  if (USING_CUDA)
    if (CMAKE_CXX_COMPILER_ID MATCHES "nvcc")
      enable_language(CUDA)
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
