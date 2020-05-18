cmake_minimum_required(VERSION 3.10)

set(FILES
  in.lj)

foreach(FILE ${FILES})
  configure_file(${FILE} ${FILE} COPYONLY)
endforeach()

add_subdirectory(snap)
