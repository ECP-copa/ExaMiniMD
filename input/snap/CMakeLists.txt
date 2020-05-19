set(FILES
  in.snap.Ta06A
  in.snap.W
  Ta06A.snap
  Ta06A.snapcoeff
  Ta06A.snapparam
  W.snapcoeff
  W.snapparam)

foreach(FILE ${FILES})
  configure_file(${FILE} ${FILE} COPYONLY)
endforeach()
