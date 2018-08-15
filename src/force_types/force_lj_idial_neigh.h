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
      if( (strcmp(argv[i+1], "NEIGH_FULL") == 0) )
        force_iteration_type = FORCE_ITER_NEIGH_FULL;
      if( (strcmp(argv[i+1], "NEIGH_HALF") == 0) ) {
        force_iteration_type = FORCE_ITER_NEIGH_HALF;
      }
#endif
#ifdef FORCE_MODULES_INSTANTIATION
    else if (input->force_type == FORCE_LJ_IDIAL) {
      bool half_neigh = input->force_iteration_type == FORCE_ITER_NEIGH_HALF;
      switch ( input->neighbor_type ) {
        #define FORCETYPE_ALLOCATION_MACRO(NeighType)  ForceLJIDialNeigh<NeighType>(input->input_data.words[input->force_line],system,half_neigh)
        #include <modules_neighbor.h>
        #undef FORCETYPE_ALLOCATION_MACRO
      }
    }
#endif


#if !defined(MODULES_OPTION_CHECK) && \
    !defined(FORCE_MODULES_INSTANTIATION)

#ifndef FORCE_LJ_IDIAL_NEIGH_H
#define FORCE_LJ_IDIAL_NEIGH_H
#include<force.h>

template<class NeighborClass>
class ForceLJIDialNeigh: public Force {
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
  t_fparams lj1,lj2,cutsq,intensity;
  t_fparams_rnd rnd_lj1,rnd_lj2,rnd_cutsq,rnd_intensity;

  typedef typename NeighborClass::t_neigh_list t_neigh_list;
  t_neigh_list neigh_list;

public:
  struct TagFullNeigh {};
  struct TagHalfNeigh {};

  typedef Kokkos::RangePolicy<TagFullNeigh,Kokkos::IndexType<T_INT> > t_policy_full_neigh;
  typedef Kokkos::RangePolicy<TagHalfNeigh,Kokkos::IndexType<T_INT> > t_policy_half_neigh;

  ForceLJIDialNeigh (char** args, System* system, bool half_neigh_);

  void init_coeff(int nargs, char** args);

  void compute(System* system, Binning* binning, Neighbor* neighbor );

  KOKKOS_INLINE_FUNCTION
  void operator() (TagFullNeigh, const T_INT& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagHalfNeigh, const T_INT& i) const;

  const char* name();
};

#define FORCE_MODULES_EXTERNAL_TEMPLATE
#define FORCETYPE_DECLARE_TEMPLATE_MACRO(NeighType) ForceLJIDialNeigh<NeighType>
#include<modules_neighbor.h>
#undef FORCETYPE_DECLARE_TEMPLATE_MACRO
#undef FORCE_MODULES_EXTERNAL_TEMPLATE
#endif
#endif
