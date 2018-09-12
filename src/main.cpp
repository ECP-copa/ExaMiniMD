//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimer in the documentation
//       and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//************************************************************************

#include<examinimd.h>

// ExaMiniMD can be used as a library
// This main file is simply a driver

#ifdef EXAMINIMD_ENABLE_MPI
#include "mpi.h"
#endif
#include <Kokkos_RemoteSpaces.hpp>

int main(int argc, char* argv[]) {

   #ifdef EXAMINIMD_ENABLE_MPI
   MPI_Init(&argc,&argv);
   #endif
   #ifdef KOKKOS_ENABLE_NVSHMEM
   shmemx_init_attr_t attr;
   auto mpi_comm = MPI_COMM_WORLD;
   attr.mpi_comm = &mpi_comm;
   shmemx_init_attr (SHMEMX_INIT_WITH_MPI_COMM, &attr);
   #endif
   #ifdef KOKKOS_ENABLE_SHMEMSPACE
   shmem_init();
   #endif
   #endif

   Kokkos::initialize(argc,argv);

   ExaMiniMD examinimd;
   examinimd.init(argc,argv);
  
   examinimd.run(examinimd.input->nsteps);

   //   examinimd.check_correctness();

   examinimd.print_performance();

   examinimd.shutdown();

   Kokkos::finalize();

   #ifdef EXAMINIMD_ENABLE_MPI
   //MPI_Finalize();
   #endif
}

