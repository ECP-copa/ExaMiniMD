#ifndef BINNING_KKSORT_H
#define BINNING_KKSORT_H
#include<binning.h>
#include<Kokkos_Sort.hpp>

class BinningKKSort: public Binning {
  typedef Kokkos::BinOp3D<t_x_const> t_binop;
  typedef Kokkos::BinSort<t_x_const,t_binop,Kokkos::DefaultExecutionSpace,T_INT> t_sorter;
  t_sorter sorter;

public:
  BinningKKSort(System* s);
  void create_binning(T_X_FLOAT dx, T_X_FLOAT dy, T_X_FLOAT dz, int halo_depth, bool do_local, bool do_ghost, bool sort);
};
#endif
