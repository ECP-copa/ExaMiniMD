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

#ifdef MODULES_OPTION_CHECK
   if( (strcmp(argv[i+1], "MPI") == 0) )
     comm_type = COMM_MPI;
#endif
#ifdef COMM_MODULES_INSTANTIATION
   else if(input->comm_type == COMM_MPI) {
     comm = new CommMPI(system,input->force_cutoff + input->neighbor_skin);
   }
#endif


#if !defined(MODULES_OPTION_CHECK) && !defined(COMM_MODULES_INSTANTIATION)
#ifndef COMM_MPI_H
#define COMM_MPI_H
#include<comm.h>

#ifndef EXAMINIMD_ENABLE_MPI
#error "Trying to compile CommMPI without MPI"
#endif

#include "mpi.h"

class CommMPI: public Comm {

  // Variables Comm doesn't own but requires for computations

  T_INT N_local;
  T_INT N_ghost;

  System s;

  // Owned Variables

  int phase; // Communication Phase
  int proc_neighbors_recv[6]; // Neighbor for each phase
  int proc_neighbors_send[6]; // Neighbor for each phase
  int proc_num_recv[6];  // Number of received particles in each phase
  int proc_num_send[6];  // Number of send particles in each phase
  int proc_pos[3];       // My process position
  int proc_grid[3];      // Process Grid size
  int proc_rank;         // My Process rank
  int proc_size;         // Number of processes

  T_INT num_ghost[6];
  T_INT ghost_offsets[6];

  T_INT num_packed;
  Kokkos::View<int, Kokkos::MemoryTraits<Kokkos::Atomic> > pack_count;
  Kokkos::View<Particle*> pack_buffer;
  Kokkos::View<Particle*> unpack_buffer;
  typedef Kokkos::View<T_X_FLOAT*[3], Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Unmanaged>> t_buffer_update;
  t_buffer_update pack_buffer_update;
  t_buffer_update unpack_buffer_update;

  Kokkos::View<T_INT**,Kokkos::LayoutRight> pack_indicies_all;
  Kokkos::View<T_INT*,Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Unmanaged> > pack_indicies;
  Kokkos::View<T_INT*,Kokkos::LayoutRight > exchange_dest_list;

public:

  struct TagUnpack {};

  struct TagExchangeSelf {};
  struct TagExchangePack {};
  struct TagExchangeCreateDestList {};
  struct TagExchangeCompact {};
  
  struct TagHaloSelf {};
  struct TagHaloPack {};
  struct TagHaloUpdateSelf {};
  struct TagHaloUpdatePack {};
  struct TagHaloUpdateUnpack {};

  struct TagHaloForceSelf {};
  struct TagHaloForcePack {};
  struct TagHaloForceUnpack {};

  struct TagPermuteIndexList {};

  struct TagCreateGlobalIndecies {};

  CommMPI(System* s, T_X_FLOAT comm_depth_);
  void init();
  void create_domain_decomposition();
  void permute_index_lists(Binning::t_permute_vector& permute_vector);
  void exchange();
  void exchange_halo();
  void update_halo();
  void update_force();
  void scan_int(T_INT* vals, T_INT count);
  void reduce_int(T_INT* vals, T_INT count);
  void reduce_float(T_FLOAT* vals, T_INT count);
  void reduce_max_int(T_INT* vals, T_INT count);
  void reduce_max_float(T_FLOAT* vals, T_INT count);
  void reduce_min_int(T_INT* vals, T_INT count);
  void reduce_min_float(T_FLOAT* vals, T_INT count);

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagExchangeSelf, 
                   const T_INT& i) const {
    if(proc_grid[0]==1) {
      const T_X_FLOAT x = s.x(i,0);
      if(x>s.domain_x) s.x(i,0) -= s.domain_x;
      if(x<0)          s.x(i,0) += s.domain_x;
    }
    if(proc_grid[1]==1) {
      const T_X_FLOAT y = s.x(i,1);
      if(y>s.domain_y) s.x(i,1) -= s.domain_y;
      if(y<0)          s.x(i,1) += s.domain_y;
    }
    if(proc_grid[2]==1) {
      const T_X_FLOAT z = s.x(i,2);
      if(z>s.domain_z) s.x(i,2) -= s.domain_z;
      if(z<0)          s.x(i,2) += s.domain_z;
    }   
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagExchangePack, 
                   const T_INT& i) const {
    if(s.type(i)<0) return;
    if( (phase == 0) && (s.x(i,0)>s.sub_domain_hi_x)) {
      const int pack_idx = pack_count()++;
      if( pack_idx < pack_indicies.extent(0) ) {
        pack_indicies(pack_idx) = i; 
        Particle p = s.get_particle(i);
        s.type(i) = -1;
        if(proc_pos[0] == proc_grid[0]-1)
          p.x -= s.domain_x;
        pack_buffer(pack_idx) = p;
      }
    }  

    if( (phase == 1) && (s.x(i,0)<s.sub_domain_lo_x)) {
      const int pack_idx = pack_count()++;
      if( pack_idx < pack_indicies.extent(0) ) {
        pack_indicies(pack_idx) = i;
        Particle p = s.get_particle(i);
        s.type(i) = -1;
        if(proc_pos[0] == 0)
          p.x += s.domain_x;
        pack_buffer(pack_idx) = p;
      }
    }

    if( (phase == 2) && (s.x(i,1)>s.sub_domain_hi_y)) {
      const int pack_idx = pack_count()++;
      if( pack_idx < pack_indicies.extent(0) ) {
        pack_indicies(pack_idx) = i;
        Particle p = s.get_particle(i);
        s.type(i) = -1;
        if(proc_pos[1] == proc_grid[1]-1)
          p.y -= s.domain_y;
        pack_buffer(pack_idx) = p;
      }
    }
    if( (phase == 3) && (s.x(i,1)<s.sub_domain_lo_y)) {
      const int pack_idx = pack_count()++;
      if( pack_idx < pack_indicies.extent(0) ) {
        pack_indicies(pack_idx) = i;
        Particle p = s.get_particle(i);
        s.type(i) = -1;
        if(proc_pos[1] == 0)
          p.y += s.domain_y;
        pack_buffer(pack_idx) = p;
      }
    }

    if( (phase == 4) && (s.x(i,2)>s.sub_domain_hi_z)) {
      const int pack_idx = pack_count()++;
      if( pack_idx < pack_indicies.extent(0) ) {
        pack_indicies(pack_idx) = i;
        Particle p = s.get_particle(i);
        s.type(i) = -1;
        if(proc_pos[2] == proc_grid[2]-1)
          p.z -= s.domain_z;
        pack_buffer(pack_idx) = p;
      }
    }
    if( (phase == 5) && (s.x(i,2)<s.sub_domain_lo_z)) {
      const int pack_idx = pack_count()++;
      if( pack_idx < pack_indicies.extent(0) ) {
        pack_indicies(pack_idx) = i;
        Particle p = s.get_particle(i);
        s.type(i) = -1;
        if(proc_pos[2] == 0)
          p.z += s.domain_z;
        pack_buffer(pack_idx) = p;
      }
    }
  } 

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagExchangeCreateDestList,
                   const T_INT& i, T_INT& c, const bool final) const {
    if(s.type(i)<0) {
      if(final) {
        exchange_dest_list(c) = i;
      }
      c++;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagExchangeCompact,
                   const T_INT& ii, T_INT& c, const bool final) const {
    const T_INT i = N_local+N_ghost-1-ii;
    if(s.type(i)>=0) {
      if(final) {
        s.copy(exchange_dest_list(c),i,0,0,0);
      }
      c++;
    }
  }

  
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloSelf,
                   const T_INT& i) const {
    if(phase == 0) {
      if( s.x(i,0)>=s.sub_domain_hi_x - comm_depth ) {
        const int pack_idx = pack_count()++;
        if((pack_idx < pack_indicies.extent(0)) && (N_local+N_ghost+pack_idx< s.x.extent(0))) {
          pack_indicies(pack_idx) = i;
          Particle p = s.get_particle(i);
          p.x -= s.domain_x;
          s.set_particle(N_local + N_ghost + pack_idx, p);
        }
      }
    }
    if(phase == 1) {
      if( s.x(i,0)<=s.sub_domain_lo_x + comm_depth ) {
        const int pack_idx = pack_count()++;
        if((pack_idx < pack_indicies.extent(0)) && (N_local+N_ghost+pack_idx< s.x.extent(0))) {
          pack_indicies(pack_idx) = i;
          Particle p = s.get_particle(i);
          p.x += s.domain_x;
          s.set_particle(N_local + N_ghost + pack_idx, p);
        }
      }
    }
    if(phase == 2) {
      if( s.x(i,1)>=s.sub_domain_hi_y - comm_depth ) {
        const int pack_idx = pack_count()++;
        if((pack_idx < pack_indicies.extent(0)) && (N_local+N_ghost+pack_idx< s.x.extent(0))) {
          pack_indicies(pack_idx) = i;
          Particle p = s.get_particle(i);
          p.y -= s.domain_y;
          s.set_particle(N_local + N_ghost + pack_idx, p);
        }
      }
    }
    if(phase == 3) {
      if( s.x(i,1)<=s.sub_domain_lo_y + comm_depth ) {
        const int pack_idx = pack_count()++;
        if((pack_idx < pack_indicies.extent(0)) && (N_local+N_ghost+pack_idx< s.x.extent(0))) {
          pack_indicies(pack_idx) = i;
          Particle p = s.get_particle(i);
          p.y += s.domain_y;
          s.set_particle(N_local + N_ghost + pack_idx, p);
        }
      }
    }
    if(phase == 4) {
      if( s.x(i,2)>=s.sub_domain_hi_z - comm_depth ) {
        const int pack_idx = pack_count()++;
        if((pack_idx < pack_indicies.extent(0)) && (N_local+N_ghost+pack_idx< s.x.extent(0))) {
          pack_indicies(pack_idx) = i;
          Particle p = s.get_particle(i);
          p.z -= s.domain_z;
          s.set_particle(N_local + N_ghost + pack_idx, p);
        }
      }
    }
    if(phase == 5) {
      if( s.x(i,2)<=s.sub_domain_lo_z + comm_depth ) {
        const int pack_idx = pack_count()++;
        if((pack_idx < pack_indicies.extent(0)) && (N_local+N_ghost+pack_idx< s.x.extent(0))) {
          pack_indicies(pack_idx) = i;
          Particle p = s.get_particle(i);
          p.z += s.domain_z;
          s.set_particle(N_local + N_ghost + pack_idx, p);
        }
      }
    }

  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloPack,
                   const T_INT& i) const {
    if(phase == 0) {
      if( s.x(i,0)>=s.sub_domain_hi_x - comm_depth ) {
        const int pack_idx = pack_count()++;
        if(pack_idx < pack_indicies.extent(0)) {
          pack_indicies(pack_idx) = i;
          Particle p = s.get_particle(i);
          if(proc_pos[0] == proc_grid[0]-1)
            p.x -= s.domain_x;
          pack_buffer(pack_idx) = p;
        }
      }
    }
    if(phase == 1) {
      if( s.x(i,0)<=s.sub_domain_lo_x + comm_depth ) {
        const int pack_idx = pack_count()++;
        if(pack_idx < pack_indicies.extent(0)) {
          pack_indicies(pack_idx) = i;
          Particle p = s.get_particle(i);
          if(proc_pos[0] == 0)
            p.x += s.domain_x;
          pack_buffer(pack_idx) = p;
        }
      }
    }
    if(phase == 2) {
      if( s.x(i,1)>=s.sub_domain_hi_y - comm_depth ) {
        const int pack_idx = pack_count()++;
        if(pack_idx < pack_indicies.extent(0)) {
          pack_indicies(pack_idx) = i;
          Particle p = s.get_particle(i);
          if(proc_pos[1] == proc_grid[1]-1)
            p.y -= s.domain_y;
          pack_buffer(pack_idx) = p;
        }
      }
    }
    if(phase == 3) {
      if( s.x(i,1)<=s.sub_domain_lo_y + comm_depth ) {
        const int pack_idx = pack_count()++;
        if(pack_idx < pack_indicies.extent(0)) {
          pack_indicies(pack_idx) = i;
          Particle p = s.get_particle(i);
          if(proc_pos[1] == 0)
            p.y += s.domain_y;
          pack_buffer(pack_idx) = p;
        }
      }
    }
    if(phase == 4) {
      if( s.x(i,2)>=s.sub_domain_hi_z - comm_depth ) {
        const int pack_idx = pack_count()++;
        if(pack_idx < pack_indicies.extent(0)) {
          pack_indicies(pack_idx) = i;
          Particle p = s.get_particle(i);
          if(proc_pos[2] == proc_grid[2]-1)
            p.z -= s.domain_z;
          pack_buffer(pack_idx) = p;
        }
      }
    }
    if(phase == 5) {
      if( s.x(i,2)<=s.sub_domain_lo_z + comm_depth ) {
        const int pack_idx = pack_count()++;
        if(pack_idx < pack_indicies.extent(0)) {
          pack_indicies(pack_idx) = i;
          Particle p = s.get_particle(i);
          if(proc_pos[2] == 0)
            p.z += s.domain_z;
          pack_buffer(pack_idx) = p;
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagUnpack,
                   const T_INT& i) const {
    s.set_particle(N_local+N_ghost+i, unpack_buffer(i));
  }


  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloUpdateSelf,
                   const T_INT& ii) const {

    const T_INT i = pack_indicies(ii);
    T_X_FLOAT x_i = s.x(i,0);
    T_X_FLOAT y_i = s.x(i,1);
    T_X_FLOAT z_i = s.x(i,2);

    switch (phase) {
      case 0: x_i -= s.domain_x; break;
      case 1: x_i += s.domain_x; break;
      case 2: y_i -= s.domain_y; break;
      case 3: y_i += s.domain_y; break;
      case 4: z_i -= s.domain_z; break;
      case 5: z_i += s.domain_z; break;
    }
    s.x(N_local + N_ghost + ii, 0) = x_i;
    s.x(N_local + N_ghost + ii, 1) = y_i;
    s.x(N_local + N_ghost + ii, 2) = z_i;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloUpdatePack,
                   const T_INT& ii) const {

    const T_INT i = pack_indicies(ii);
    T_X_FLOAT x_i = s.x(i,0);
    T_X_FLOAT y_i = s.x(i,1);
    T_X_FLOAT z_i = s.x(i,2);

    switch (phase) {
      case 0: if(proc_pos[0] == proc_grid[0]-1) x_i -= s.domain_x; break;
      case 1: if(proc_pos[0] == 0)              x_i += s.domain_x; break;
      case 2: if(proc_pos[1] == proc_grid[1]-1) y_i -= s.domain_y; break;
      case 3: if(proc_pos[1] == 0)              y_i += s.domain_y; break;
      case 4: if(proc_pos[2] == proc_grid[2]-1) z_i -= s.domain_z; break;
      case 5: if(proc_pos[2] == 0)              z_i += s.domain_z; break;
    }
    pack_buffer_update(ii, 0) = x_i;
    pack_buffer_update(ii, 1) = y_i;
    pack_buffer_update(ii, 2) = z_i;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloUpdateUnpack,
                   const T_INT& ii) const {
    s.x(N_local + N_ghost + ii, 0) = unpack_buffer_update(ii, 0);
    s.x(N_local + N_ghost + ii, 1) = unpack_buffer_update(ii, 1);
    s.x(N_local + N_ghost + ii, 2) = unpack_buffer_update(ii, 2);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloForceSelf,
                   const T_INT& ii) const {

    const T_INT i = pack_indicies(ii);
    T_F_FLOAT fx_i = s.f(ghost_offsets[phase] + ii,0);
    T_F_FLOAT fy_i = s.f(ghost_offsets[phase] + ii,1);
    T_F_FLOAT fz_i = s.f(ghost_offsets[phase] + ii,2);

    s.f(i, 0) += fx_i;
    s.f(i, 1) += fy_i;
    s.f(i, 2) += fz_i;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloForcePack,
                   const T_INT& ii) const {

    T_F_FLOAT fx_i = s.f(ghost_offsets[phase] + ii,0);
    T_F_FLOAT fy_i = s.f(ghost_offsets[phase] + ii,1);
    T_F_FLOAT fz_i = s.f(ghost_offsets[phase] + ii,2);

    unpack_buffer_update(ii, 0) = fx_i;
    unpack_buffer_update(ii, 1) = fy_i;
    unpack_buffer_update(ii, 2) = fz_i;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagHaloForceUnpack,
                   const T_INT& ii) const {

    const T_INT i = pack_indicies(ii);

    s.f(i, 0) += pack_buffer_update(ii, 0);
    s.f(i, 1) += pack_buffer_update(ii, 1);
    s.f(i, 2) += pack_buffer_update(ii, 2);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagCreateGlobalIndecies,
                   const T_INT& i) const {
    s.global_index(i) = N_MAX_MASK * proc_rank + i; 
  }

  const char* name();
  int process_rank();
  int num_processes();
  void error(const char *);
};

#endif
#endif // MODULES_OPTION_CHECK

