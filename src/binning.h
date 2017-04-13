#ifndef BINNING_H
#define BINNING_H
#include <types.h>
#include <system.h>
class Binning {
protected:
  System* system;


public: 

  T_INT nbinx, nbiny, nbinz, nhalo;
  T_X_FLOAT minx,maxx,miny,maxy,minz,maxz;

  typedef Kokkos::View<int***> t_bincount;
  typedef Kokkos::View<T_INT***> t_binoffsets;
  typedef Kokkos::View<T_INT*> t_permute_vector;

  t_bincount bincount;
  t_binoffsets binoffsets;
  t_permute_vector permute_vector;

  bool is_sorted;

  Binning(System* s);
  
  virtual void create_binning(T_X_FLOAT dx, T_X_FLOAT dy, T_X_FLOAT dz, int halo_depth, bool do_local, bool do_ghost, bool sort);
};

#include<modules_binning.h>

#endif
