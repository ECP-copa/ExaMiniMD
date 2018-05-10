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

#ifndef SYSTEM_H
#define SYSTEM_H
#include<types.h>

struct Particle {
  KOKKOS_INLINE_FUNCTION
  Particle() {
    x=y=z=vx=vy=vz=mass=q=0.0;
    id = type = 0;
    global_index = 0;
  }

  T_X_FLOAT x,y,z;
  T_V_FLOAT vx,vy,vz,mass;
  T_FLOAT q;
  T_INT id;
  T_INDEX global_index;
  int type;
};

class System {
public:
  T_INT N;       // Number of Global Particles
  T_INT N_max;   // Number of Particles I could have in available storage
  T_INT N_local; // Number of owned Particles
  T_INT N_ghost; // Number of non-owned Particles

  int ntypes;

  // Per Particle Property
  t_x x;         // Positions
  t_v v;         // Velocities
  t_f f;         // Forces

  t_type  type;   // Particle Type
  t_id    id;     // Particle ID
  t_index global_index; // Index for PGAS indexing  
  
  t_q q;         // Charge

  t_x_shmem x_shmem; 

  // Per Type Property
  t_mass mass;

  // Simulation domain
  T_X_FLOAT domain_x, domain_y, domain_z;

  // Simulation sub domain (for example of a single MPI rank)
  T_X_FLOAT sub_domain_x, sub_domain_y, sub_domain_z;
  T_X_FLOAT sub_domain_lo_x, sub_domain_lo_y, sub_domain_lo_z;
  T_X_FLOAT sub_domain_hi_x, sub_domain_hi_y, sub_domain_hi_z;

  // Units
  T_FLOAT boltz,mvv2e,dt;

  // Should this process print messages
  bool do_print;
  // Should print LAMMPS style messages
  bool print_lammps;

  System();
  ~System() {};
  void init();
  void destroy();

  void grow(T_INT new_N);

  KOKKOS_INLINE_FUNCTION
  Particle get_particle(const T_INT& i) const {
    Particle p;
    p.x  = x(i,0); p.y  = x(i,1); p.z  = x(i,2);
    p.vx = v(i,0); p.vy = v(i,1); p.vz = v(i,2);
    p.q = q(i);
    p.id = id(i);
    p.type = type(i);
    p.global_index = global_index(i);
    return p;
  }

  KOKKOS_INLINE_FUNCTION
  void set_particle(const T_INT& i, const Particle& p) const {
    x(i,0) = p.x;  x(i,1) = p.y;  x(i,2) = p.z;
    v(i,0) = p.vx; v(i,1) = p.vy; v(i,2) = p.vz;
    q(i) = p.q;
    id(i) = p.id;
    type(i) = p.type;
    global_index(i) = p.global_index;
  }

  KOKKOS_INLINE_FUNCTION
  void copy(T_INT dest, T_INT src, int nx, int ny, int nz) const {
    x(dest,0) = x(src,0) + domain_x * nx;
    x(dest,1) = x(src,1) + domain_y * ny;
    x(dest,2) = x(src,2) + domain_z * nz;
    v(dest,0) = v(src,0);
    v(dest,1) = v(src,1);
    v(dest,2) = v(src,2);
    type(dest) = type(src);
    id(dest) = id(src);
    q(dest) = q(src);
    global_index(dest) = global_index(src);
  }

  KOKKOS_INLINE_FUNCTION
  void copy_halo_update(T_INT dest, T_INT src, int nx, int ny, int nz) const {
    x(dest,0) = x(src,0) + domain_x * nx;
    x(dest,1) = x(src,1) + domain_y * ny;
    x(dest,2) = x(src,2) + domain_z * nz;
  }

  void print_particles();
};
#endif
