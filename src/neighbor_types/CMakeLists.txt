SET(HEADERS
  neighbor_2d.h
  neighbor_csr.h
  neighbor_csr_map_constr.h)

SET(SOURCES
  neighbor_2d.cpp
  neighbor_csr.cpp
  neighbor_csr_map_constr.cpp)

target_sources(ExaMiniMD PRIVATE ${SOURCES} ${HEADERS})
