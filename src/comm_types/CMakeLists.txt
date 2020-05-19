SET(HEADERS
  comm_mpi.h
  comm_serial.h)

SET(SOURCES
  comm_mpi.cpp
  comm_serial.cpp)

target_sources(ExaMiniMD PRIVATE ${SOURCES} ${HEADERS})
