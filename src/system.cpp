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

#include<system.h>
#ifdef EXAMINIMD_ENABLE_MPI
#include<mpi.h>
#endif
System::System() {
  N = 0;
  N_max = 0;
  N_local = 0;
  N_ghost = 0;
  ntypes = 1;
  x = t_x();
  v = t_v();
  f = t_f();
  id = t_id();
  global_index = t_index();
  type = t_type();
  q = t_q();
  mass = t_mass();
  domain_x = domain_y = domain_z = 0.0;
  sub_domain_x = sub_domain_y = sub_domain_z = 0.0;
  sub_domain_hi_x = sub_domain_hi_y = sub_domain_hi_z = 0.0;
  sub_domain_lo_x = sub_domain_lo_y = sub_domain_lo_z = 0.0;
  mvv2e = boltz = dt = 0.0;
#ifdef EXAMINIMD_ENABLE_MPI
  int proc_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  do_print = proc_rank == 0;
#else
  do_print = true;
#endif
  print_lammps = false;
}

void System::init() {
  x = t_x("System::x",N_max);
  v = t_v("System::v",N_max);
  f = t_f("System::f",N_max);
  id = t_id("System::id",N_max);
  global_index = t_index("System::global_index",N_max);
  type = t_type("System::type",N_max);
  q = t_q("System::q",N_max);
  mass = t_mass("System::mass",ntypes);
}

void System::destroy() {
  N_max = 0;
  N_local = 0;
  N_ghost = 0;
  ntypes = 1;
  x = t_x();
  v = t_v();
  f = t_f();
  id = t_id();
  global_index = t_index();
  type = t_type();
  q = t_q();
  mass = t_mass();
}

void System::grow(T_INT N_new) {
  if(N_new > N_max) {
    N_max = N_new; // Number of global Particles

    Kokkos::resize(x,N_max);      // Positions
    Kokkos::resize(v,N_max);      // Velocities
    Kokkos::resize(f,N_max);      // Forces

    Kokkos::resize(id,N_max);     // Id
    Kokkos::resize(global_index,N_max);     // Id

    Kokkos::resize(type,N_max);   // Particle Type

    Kokkos::resize(q,N_max);      // Charge
{
#ifdef EXAMINIMD_ENABLE_MPI
  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
#else
  int num_ranks = 1;
#endif
    int* rank_list = new int[num_ranks];
    for(int i=0; i<num_ranks; i++)
      rank_list[i] = i;
    Kokkos::DefaultRemoteMemorySpace space;
    x_shmem = Kokkos::allocate_symmetric_remote_view<t_x_shmem>("X_shmem",num_ranks,rank_list,N_max); 
}

  }
}

void System::print_particles() {
  printf("Print all particles: \n");
  printf("  Owned: %d\n",N_local);
  for(T_INT i=0;i<N_local;i++) {
    printf("    %d %lf %lf %lf | %lf %lf %lf | %lf %lf %lf | %d %e\n",i,
        double(x(i,0)),double(x(i,1)),double(x(i,2)),
        double(v(i,0)),double(v(i,1)),double(v(i,2)),
        double(f(i,0)),double(f(i,1)),double(f(i,2)),
        type(i),q(i)
        );
  }

  printf("  Ghost: %d\n",N_ghost);
  for(T_INT i=N_local;i<N_local+N_ghost;i++) {
    printf("    %d %lf %lf %lf | %lf %lf %lf | %lf %lf %lf | %d %e\n",i,
        double(x(i,0)),double(x(i,1)),double(x(i,2)),
        double(v(i,0)),double(v(i,1)),double(v(i,2)),
        double(f(i,0)),double(f(i,1)),double(f(i,2)),
        type(i),q(i)
        );
  }

}
