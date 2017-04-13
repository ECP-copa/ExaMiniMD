#include<examinimd.h>

// ExaMiniMD can be used as a library
// This main file is simply a driver

#ifdef EXAMINIMD_ENABLE_MPI
#include "mpi.h"
#endif

int main(int argc, char* argv[]) {

   #ifdef EXAMINIMD_ENABLE_MPI
   MPI_Init(&argc,&argv);
   #endif

   Kokkos::initialize(argc,argv);

   ExaMiniMD examinimd;
   examinimd.init(argc,argv);
  
   examinimd.run(examinimd.input->nsteps);

   examinimd.check_correctness();

   examinimd.print_performance();

   examinimd.shutdown();

   Kokkos::finalize();

   #ifdef EXAMINIMD_ENABLE_MPI
   MPI_Finalize();
   #endif
}

