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

/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#ifdef MODULES_OPTION_CHECK
      if( (strcmp(argv[i+1], "NEIGH_FULL") == 0) )
        force_iteration_type = FORCE_ITER_NEIGH_FULL;
#endif
#ifdef FORCE_MODULES_INSTANTIATION
    else if (input->force_type == FORCE_SNAP) {
      bool half_neigh = input->force_iteration_type == FORCE_ITER_NEIGH_HALF;
      if(half_neigh) Kokkos::abort("ForceSNAP does not support half neighborlist");
      switch ( input->neighbor_type ) {
        #define FORCETYPE_ALLOCATION_MACRO(NeighType)  ForceSNAP<NeighType>(input->input_data.words[input->force_line],system,half_neigh)
        #include <modules_neighbor.h>
        #undef FORCETYPE_ALLOCATION_MACRO
      }
    }

#endif


#if !defined(MODULES_OPTION_CHECK) && \
    !defined(FORCE_MODULES_INSTANTIATION)

#ifndef FORCE_SNAP_NEIGH_H
#define FORCE_SNAP_NEIGH_H
#include<force.h>
#include<sna.h>

template<class NeighborClass>
class ForceSNAP : public Force {
public:
  // lammps splice
// Routines for both the CPU and GPU backend
template<int NEIGHFLAG, int EVFLAG>
struct TagComputeForce{};


// GPU backend only
struct TagComputeNeigh{};
struct TagComputeCayleyKlein{};
struct TagPreUi{};
struct TagComputeUiSmall{}; // more parallelism, more divergence
struct TagComputeUiLarge{}; // less parallelism, no divergence
struct TagTransformUi{}; // re-order ulisttot from SoA to AoSoA, zero ylist
struct TagComputeZi{};
struct TagBeta{};
struct TagComputeBi{};
struct TagTransformBi{}; // re-order blist from AoSoA to AoS
struct TagComputeYi{};
struct TagComputeYiWithZlist{};
template<int dir>
struct TagComputeFusedDeidrjSmall{}; // more parallelism, more divergence
template<int dir>
struct TagComputeFusedDeidrjLarge{}; // less parallelism, no divergence

// CPU backend only
struct TagComputeNeighCPU{};
struct TagPreUiCPU{};
struct TagComputeUiCPU{};
struct TagTransformUiCPU{};
struct TagComputeZiCPU{};
struct TagBetaCPU{};
struct TagComputeBiCPU{};
struct TagZeroYiCPU{};
struct TagComputeYiCPU{};
struct TagComputeDuidrjCPU{};
struct TagComputeDeidrjCPU{};

  ForceSNAP(char** args, System* system, bool half_neigh_);
  ~ForceSNAP();
  
  void init_coeff(int nargs, char** args);
  void compute(System* system, Binning* binning, Neighbor* neighbor );

  const char* name() {return "ForceSNAP";}
// lammps splice
/*  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeForce<NEIGHFLAG,EVFLAG>,const int& ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagBetaCPU,const int& ii) const;
*/

  // GPU backend only
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeNeigh,const typename Kokkos::TeamPolicy<TagComputeNeigh>::member_type& team) const;

/*
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeCayleyKlein, const int iatom_mod, const int jnbor, const int iatom_div) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPreUi,const int iatom_mod, const int j, const int iatom_div) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeUiSmall,const typename Kokkos::TeamPolicyTagComputeUiSmall>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeUiLarge,const typename Kokkos::TeamPolicyTagComputeUiLarge>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagTransformUi,const int iatom_mod, const int j, const int iatom_div) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeZi,const int iatom_mod, const int idxz, const int iatom_div) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagBeta, const int& ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeBi,const int iatom_mod, const int idxb, const int iatom_div) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagTransformBi,const int iatom_mod, const int idxb, const int iatom_div) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeYi,const int iatom_mod, const int idxz, const int iatom_div) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeYiWithZlist,const int iatom_mod, const int idxz, const int iatom_div) const;

  template<int dir>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeFusedDeidrjSmall<dir>,const typename Kokkos::TeamPolicyTagComputeFusedDeidrjSmall<dir> >::member_type& team) const;

  template<int dir>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeFusedDeidrjLarge<dir>,const typename Kokkos::TeamPolicyTagComputeFusedDeidrjLarge<dir> >::member_type& team) const;

  // CPU backend only
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeNeighCPU,const typename Kokkos::TeamPolicyTagComputeNeighCPU>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPreUiCPU,const typename Kokkos::TeamPolicyTagPreUiCPU>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeUiCPU,const typename Kokkos::TeamPolicyTagComputeUiCPU>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagTransformUiCPU, const int j, const int iatom) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeZiCPU,const int& ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeBiCPU,const typename Kokkos::TeamPolicyTagComputeBiCPU>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeYiCPU,const int& ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeDuidrjCPU,const typename Kokkos::TeamPolicyTagComputeDuidrjCPU>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeDeidrjCPU,const typename Kokkos::TeamPolicyTagComputeDeidrjCPU>::member_type& team) const;
*/

protected:
  System* system;

  typedef typename NeighborClass::t_neigh_list t_neigh_list;
  t_neigh_list neigh_list;

  int ncoeff, ncoeffq, ncoeffall;
  typedef Kokkos::View<T_F_FLOAT**> t_bvec;
  t_bvec bvec;
  typedef Kokkos::View<T_F_FLOAT***> t_dbvec;
  t_dbvec dbvec;
  SNA sna;

  int nmax,max_neighs;
   
  int chunk_size,chunk_offset,chunksize;
  int host_flag;

  // How many interactions can be run concurrently
  int concurrent_interactions;

  void allocate();
  void read_files(char *, char *);
  /*inline int equal(double* x,double* y);
  inline double dist2(double* x,double* y);
  double extra_cutoff();
  void load_balance();
  void set_sna_to_shared(int snaid,int i);
  void build_per_atom_arrays();*/

  int schedule_user;
  double schedule_time_guided;
  double schedule_time_dynamic;

  int ncalls_neigh;
  int do_load_balance;
  int ilistmask_max;
  Kokkos::View<T_INT*> ilistmast;
  int ghostinum;
  int ghostilist_max;
  Kokkos::View<T_INT*> ghostilist;
  int ghostnumneigh_max;
  Kokkos::View<T_INT*> ghostnumneigh;
  Kokkos::View<T_INT*> ghostneighs;
  Kokkos::View<T_INT*> ghostfirstneigh;
  int ghostneighs_total;
  int ghostneighs_max;

  int use_optimized;
  int use_shared_arrays;

  int i_max;
  int i_neighmax;
  int i_numpairs;
  Kokkos::View<T_INT**, Kokkos::LayoutRight> i_pairs;
  Kokkos::View<T_INT***, Kokkos::LayoutRight> i_rij;
  Kokkos::View<T_INT**, Kokkos::LayoutRight> i_inside;
  Kokkos::View<T_F_FLOAT**, Kokkos::LayoutRight> i_wj;
  Kokkos::View<T_F_FLOAT***, Kokkos::LayoutRight>i_rcutij;
  Kokkos::View<T_INT*> i_ninside;
  Kokkos::View<T_F_FLOAT****, Kokkos::LayoutRight> i_uarraytot_r, i_uarraytot_i;
  Kokkos::View<T_F_FLOAT******, Kokkos::LayoutRight> i_zarray_r, i_zarray_i;

#ifdef TIMING_INFO
  //  timespec starttime, endtime;
  double timers[4];
#endif

  double rcutmax;               // max cutoff for all elements
  int nelements;                // # of unique elements
  char **elements;              // names of unique elements
  Kokkos::View<T_F_FLOAT*> radelem;              // element radii
  Kokkos::View<T_F_FLOAT*> wjelem;               // elements weights
  Kokkos::View<T_F_FLOAT**, Kokkos::LayoutRight> coeffelem;           // element bispectrum coefficients
  Kokkos::View<T_INT*> map;                     // mapping from atom types to elements
  int twojmax, switchflag, bzeroflag, bnormflag, quadraticflag;
  int chemflag, wselfallflag, switchinnerflag, diagonalstyle;
  double rcutfac, rfac0, rmin0, wj1, wj2;
  int rcutfacflag, twojmaxflag; // flags for required parameters
  typedef Kokkos::View<T_F_FLOAT**> t_fparams;
  t_fparams cutsq;
  typedef Kokkos::View<const T_F_FLOAT**,
      Kokkos::MemoryTraits<Kokkos::RandomAccess>> t_fparams_rnd;
  t_fparams_rnd rnd_cutsq;


  t_x x;
  t_f_atomic f;
  t_type type;

public:

  KOKKOS_INLINE_FUNCTION
  void operator() (const Kokkos::TeamPolicy<>::member_type& team) const;

 // Utility routine which wraps computing per-team scratch size requirements for
  // ComputeNeigh, ComputeUi, and ComputeFusedDeidrj
  template <typename scratch_type>
  int scratch_size_helper(int values_per_team);

// Static team/tile sizes for device offload

#ifdef KOKKOS_ENABLE_HIP
  static constexpr int team_size_compute_neigh = 2;
  static constexpr int tile_size_compute_ck = 2;
  static constexpr int tile_size_pre_ui = 2;
  static constexpr int team_size_compute_ui = 2;
  static constexpr int tile_size_transform_ui = 2;
  static constexpr int tile_size_compute_zi = 2;
  static constexpr int tile_size_compute_bi = 2;
  static constexpr int tile_size_transform_bi = 2;
  static constexpr int tile_size_compute_yi = 2;
  static constexpr int team_size_compute_fused_deidrj = 2;
#else
  static constexpr int team_size_compute_neigh = 4;
  static constexpr int tile_size_compute_ck = 4;
  static constexpr int tile_size_pre_ui = 4;
  static constexpr int team_size_compute_ui = sizeof(double) == 4 ? 8 : 4;
  static constexpr int tile_size_transform_ui = 4;
  static constexpr int tile_size_compute_zi = 8;
  static constexpr int tile_size_compute_bi = 4;
  static constexpr int tile_size_transform_bi = 4;
  static constexpr int tile_size_compute_yi = 8;
  static constexpr int team_size_compute_fused_deidrj = sizeof(double) == 4 ? 4 : 2;
#endif
static constexpr int vector_length = 32;

// Custom MDRangePolicy, Rank3, to reduce verbosity of kernel launches
  // This hides the Kokkos::IndexType<int> and Kokkos::Rank<3...>
  // and reduces the verbosity of the LaunchBound by hiding the explicit
  // multiplication by vector_length
  //template <int num_tiles, class TagPairSNAP>
  //using Snap3DRangePolicy = typename Kokkos::MDRangePolicy<Kokkos::IndexType<int>, Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>, Kokkos::LaunchBounds<vector_length * num_tiles>, TagPairSNAP>;

  // Custom SnapAoSoATeamPolicy to reduce the verbosity of kernel launches
  // This hides the LaunchBounds abstraction by hiding the explicit
  // multiplication by vector length
  template <int num_teams, class TagPairSNAP>
  using SnapAoSoATeamPolicy = typename Kokkos::TeamPolicy<Kokkos::LaunchBounds<vector_length * num_teams>, TagPairSNAP>;

  Kokkos::View<T_F_FLOAT*> d_radelem;              // element radii
  Kokkos::View<T_F_FLOAT*> d_wjelem;               // elements weights
  Kokkos::View<T_F_FLOAT**, Kokkos::LayoutRight> d_coeffelem;           // element bispectrum coefficients
  Kokkos::View<T_F_FLOAT*> d_sinnerelem;           // element inner cutoff midpoint
  Kokkos::View<T_F_FLOAT*> d_dinnerelem;           // element inner cutoff half-width
  Kokkos::View<T_INT*> d_map;                    // mapping from atom types to elements
  Kokkos::View<T_INT*> d_ninside;                // ninside for all atoms in list
};




#define FORCE_MODULES_EXTERNAL_TEMPLATE
#define FORCETYPE_DECLARE_TEMPLATE_MACRO(NeighType) ForceSNAP<NeighType>
#include<modules_neighbor.h>
#undef FORCETYPE_DECLARE_TEMPLATE_MACRO
#undef FORCE_MODULES_EXTERNAL_TEMPLATE

#endif
#endif
