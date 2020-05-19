SET(HEADERS
  binning_kksort.h)

SET(SOURCES
  binning_kksort.cpp)

foreach(FILE ${SOURCES} ${HEADERS})
  target_sources(ExaMiniMD PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${FILE})
endforeach()
