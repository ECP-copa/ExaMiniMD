#ifdef MODULES_OPTION_CHECK
      if( (strcmp(argv[i+1], "SERIAL") == 0) )
        comm_type = COMM_SERIAL;
#endif
#ifdef MODULES_INSTANTIATION
      else if(input->comm_type == COMM_SERIAL) {
        comm = new CommSerial(system,input->force_cutoff + input->neighbor_skin);
      }
#endif


#if !defined(MODULES_OPTION_CHECK) && !defined(MODULES_INSTANTIATION)
#ifndef COMM_SERIAL_H
#define COMM_SERIAL_H
#include<comm.h>

class CommSerial: public Comm {

  // Variables Comm doesn't own but requires for computations

  T_INT N_local;
  T_INT N_ghost;

  System s;

  // Owned Variables

  int phase; // Communication Phase
  int num_ghost[6];

  T_INT num_packed;
  Kokkos::View<int, Kokkos::MemoryTraits<Kokkos::Atomic> > pack_count;

  Kokkos::View<T_INT**,Kokkos::LayoutRight> pack_indicies_all;
  Kokkos::View<T_INT*,Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Unmanaged> > pack_indicies;


public:

  struct TagUnpack {};

  struct TagExchangeSelf {};
  
  struct TagHaloSelf {};
  struct TagHaloUpdateSelf {};

  CommSerial(System* s, T_X_FLOAT comm_depth_);
  void exchange();
  void exchange_halo();
  void update_halo();

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagExchangeSelf, 
                   const T_INT& i) const {
    const T_X_FLOAT x = s.x(i,0);
    if(x>s.domain_x) s.x(i,0) -= s.domain_x;
    if(x<0)          s.x(i,0) += s.domain_x;

    const T_X_FLOAT y = s.x(i,1);
    if(y>s.domain_y) s.x(i,1) -= s.domain_y;
    if(y<0)          s.x(i,1) += s.domain_y;

    const T_X_FLOAT z = s.x(i,2);
    if(z>s.domain_z) s.x(i,2) -= s.domain_z;
    if(z<0)          s.x(i,2) += s.domain_z;
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
  void operator() (const TagHaloUpdateSelf,
                   const T_INT& i) const {

    Particle p = s.get_particle(pack_indicies(i));
    switch (phase) {
      case 0: p.x -= s.domain_x; break;
      case 1: p.x += s.domain_x; break;
      case 2: p.y -= s.domain_y; break;
      case 3: p.y += s.domain_y; break;
      case 4: p.z -= s.domain_z; break;
      case 5: p.z += s.domain_z; break;
    }
    s.set_particle(N_local + N_ghost + i, p);     
  }

  const char* name();
};
#endif
#endif // MODULES_OPTION_CHECK
