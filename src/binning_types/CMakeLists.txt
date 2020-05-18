cmake_minimum_required(VERSION 3.10)

SET(HEADERS
  ${HEADERS}
  ${CMAKE_CURRENT_SOURCE_DIR}/binning_kksort.h

  PARENT_SCOPE)

SET(SOURCES
  ${SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/binning_kksort.cpp

  PARENT_SCOPE)
