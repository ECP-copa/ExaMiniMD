set(HEADERS
  binning.h
  comm.h
  examinimd.h
  force.h
  input.h
  integrator.h
  integrator_nve.h
  math_extra.h
  modules_binning.h
  modules_comm.h
  modules_force.h
  modules_integrator.h
  modules_neighbor.h
  modules_property.h
  neighbor.h
  property_kine.h
  property_pote.h
  property_temperature.h
  system.h
  types.h)

set(SOURCES
  binning.cpp
  comm.cpp
  examinimd.cpp
  force.cpp
  input.cpp
  integrator.cpp
  integrator_nve.cpp
  neighbor.cpp
  property_kine.cpp
  property_pote.cpp
  property_temperature.cpp
  system.cpp)

add_executable(ExaMiniMD main.cpp ${SOURCES} ${HEADERS})

set(SUBDIRECTORIES
  binning_types
  comm_types
  force_types
  neighbor_types)

foreach(SUBDIR ${SUBDIRECTORIES})
  add_subdirectory(${SUBDIR})
endforeach()

if (USE_CUDA)
  set_target_properties(ExaMiniMD PROPERTIES LANGUAGE CUDA)
endif()

if (USE_MPI)
  target_compile_definitions(ExaMiniMD PRIVATE EXAMINIMD_ENABLE_MPI)
endif()

target_include_directories(ExaMiniMD PRIVATE ${Kokkos_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${SUBDIRECTORIES})
target_link_libraries(ExaMiniMD PRIVATE $<$<BOOL:${USE_MPI}>:MPI::MPI_CXX> Kokkos::kokkos)
