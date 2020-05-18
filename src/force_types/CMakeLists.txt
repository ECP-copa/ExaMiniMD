cmake_minimum_required(VERSION 3.10)

SET(HEADERS
  ${HEADERS}
  ${CMAKE_CURRENT_SOURCE_DIR}/force_lj_cell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/force_lj_idial_neigh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/force_lj_idial_neigh_impl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/force_lj_neigh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/force_lj_neigh_impl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/force_snap_neigh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/force_snap_neigh_impl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/sna.h

  PARENT_SCOPE)

SET(SOURCES
  ${SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/force_lj_cell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/force_lj_idial_neigh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/force_lj_neigh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/force_snap_neigh.cpp

  PARENT_SCOPE)
