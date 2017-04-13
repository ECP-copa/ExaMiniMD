#!/bin/bash

function generate_modules_header {
  echo "// Include Module header files for $1" > modules_$1.tmp
  for file in $1_*.h
  do
    echo "#include <${file}>" >> modules_$1.tmp
  done
  
  difference=`diff modules_$1.h modules_$1.tmp 2>&1 | wc -l`
echo ${difference} 
  if [ ${difference} -ne 0 ]
  then
    mv modules_$1.tmp modules_$1.h
  else
    rm modules_$1.tmp
  fi
}

generate_modules_header force
generate_modules_header comm
generate_modules_header integrator
generate_modules_header binning
generate_modules_header neighbor
generate_modules_header property