#ifndef SYSTEM_H
#define SYSTEM_H
#include<types.h>

struct Particle {
  KOKKOS_INLINE_FUNCTION
  Particle() {
    x=y=z=vx=vy=vz=mass=q=0.0;
    id = type = 0;
  }

  T_X_FLOAT x,y,z;
  T_V_FLOAT vx,vy,vz,mass;
  T_FLOAT q;
  T_INT id;
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
  t_f_duplicated f_r;         // Duplicated Forces

  t_type type;   // Particle Type
  t_id   id;     // Particle ID
    
  t_q q;         // Charge

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
    return p;
  }

  KOKKOS_INLINE_FUNCTION
  void set_particle(const T_INT& i, const Particle& p) const {
    x(i,0) = p.x;  x(i,1) = p.y;  x(i,2) = p.z;
    v(i,0) = p.vx; v(i,1) = p.vy; v(i,2) = p.vz;
    q(i) = p.q;
    id(i) = p.id;
    type(i) = p.type;
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
