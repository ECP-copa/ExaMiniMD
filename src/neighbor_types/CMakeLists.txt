cmake_minimum_required(VERSION 3.10)

SET(HEADERS
  ${HEADERS}
  ${CMAKE_CURRENT_SOURCE_DIR}/neighbor_2d.h
  ${CMAKE_CURRENT_SOURCE_DIR}/neighbor_csr.h
  ${CMAKE_CURRENT_SOURCE_DIR}/neighbor_csr_map_constr.h

  PARENT_SCOPE)

SET(SOURCES
  ${SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/neighbor_2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/neighbor_csr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/neighbor_csr_map_constr.cpp

  PARENT_SCOPE)
