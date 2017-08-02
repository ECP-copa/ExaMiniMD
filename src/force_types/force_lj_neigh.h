#ifdef MODULES_OPTION_CHECK
      if( (strcmp(argv[i+1], "NEIGH_FULL") == 0) )
        force_iteration_type = FORCE_ITER_NEIGH_FULL;
      if( (strcmp(argv[i+1], "NEIGH_HALF") == 0) ) {
        force_iteration_type = FORCE_ITER_NEIGH_HALF;
      }
#endif
#ifdef FORCE_MODULES_INSTANTIATION
    else if (input->force_type == FORCE_LJ) {
      bool half_neigh = input->force_iteration_type == FORCE_ITER_NEIGH_HALF;
      switch ( input->neighbor_type ) {
        #define FORCETYPE_ALLOCATION_MACRO(NeighType)  ForceLJNeigh<NeighType>(input->input_data.words[input->force_line],system,half_neigh)
        #include <modules_neighbor.h>
        #undef FORCETYPE_ALLOCATION_MACRO
      }
    }
#endif


#if !defined(MODULES_OPTION_CHECK) && \
    !defined(FORCE_MODULES_INSTANTIATION)

#ifndef FORCE_LJ_NEIGH_H
#define FORCE_LJ_NEIGH_H
#include<force.h>

template<class NeighborClass>
class ForceLJNeigh: public Force {
private:
  int N_local;
  t_x_const_rnd x;
  t_f f;
  t_f_atomic f_a;
  t_id id;
  t_type_const_rnd type;
  Binning::t_bincount bin_count;
  Binning::t_binoffsets bin_offsets;
  T_INT nbinx,nbiny,nbinz,nhalo;
  int step;

  typedef Kokkos::View<T_F_FLOAT**> t_fparams;
  typedef Kokkos::View<const T_F_FLOAT**,
      Kokkos::MemoryTraits<Kokkos::RandomAccess>> t_fparams_rnd;
  t_fparams lj1,lj2,cutsq;
  t_fparams_rnd rnd_lj1,rnd_lj2,rnd_cutsq;

  typedef typename NeighborClass::t_neigh_list t_neigh_list;
  t_neigh_list neigh_list;

public:
  struct TagFullNeigh {};
  struct TagHalfNeigh {};

  typedef Kokkos::RangePolicy<TagFullNeigh,Kokkos::IndexType<T_INT> > t_policy_full_neigh;
  typedef Kokkos::RangePolicy<TagHalfNeigh,Kokkos::IndexType<T_INT> > t_policy_half_neigh;

  ForceLJNeigh (char** args, System* system, bool half_neigh_);

  void init_coeff(int nargs, char** args);

  void compute(System* system, Binning* binning, Neighbor* neighbor );

  KOKKOS_INLINE_FUNCTION
  void operator() (TagFullNeigh, const T_INT& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagHalfNeigh, const T_INT& i) const;

  const char* name();
};

#define FORCE_MODULES_EXTERNAL_TEMPLATE
#define FORCETYPE_DECLARE_TEMPLATE_MACRO(NeighType) ForceLJNeigh<NeighType>
#include<modules_neighbor.h>
#undef FORCETYPE_DECLARE_TEMPLATE_MACRO
#undef FORCE_MODULES_EXTERNAL_TEMPLATE
#endif
#endif
