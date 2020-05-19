set(FILES
  in.lj)

foreach(FILE ${FILES})
  configure_file(${FILE} ${FILE} COPYONLY)
endforeach()

add_subdirectory(snap)
