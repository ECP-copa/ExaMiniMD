cmake_minimum_required(VERSION 3.10)

SET(HEADERS
  comm_mpi.h
  comm_serial.h)

SET(SOURCES
  comm_mpi.cpp
  comm_serial.cpp)

foreach(FILE ${SOURCES} ${HEADERS})
  target_sources(ExaMiniMD PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${FILE})
endforeach()
