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
  ForceSNAP(char** args, System* system, bool half_neigh_);
  ~ForceSNAP();
  
  void init_coeff(int nargs, char** args);
  void compute(System* system, Binning* binning, Neighbor* neighbor );

  const char* name() {return "ForceSNAP";}

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

  int nmax;

  // How much parallelism to use within an interaction
  int vector_length;
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
  int twojmax, diagonalstyle, switchflag, bzeroflag, quadraticflag;
  double rcutfac, rfac0, rmin0, wj1, wj2;
  int rcutfacflag, twojmaxflag; // flags for required parameters
  typedef Kokkos::View<T_F_FLOAT**> t_fparams;
  t_fparams cutsq;
  typedef Kokkos::View<const T_F_FLOAT**,
      Kokkos::MemoryTraits<Kokkos::RandomAccess>> t_fparams_rnd;
  t_fparams_rnd rnd_cutsq;


  t_x x;
  t_x_shmem x_shmem;
  t_x_shmem_local x_shmem_local;
  t_f_atomic f;
  t_type type;
  t_index global_index;

  T_X_FLOAT domain_x, domain_y, domain_z;
  int proc_rank;
public:

  struct TagForceCompute {};
  KOKKOS_INLINE_FUNCTION
  void operator() (TagForceCompute, const Kokkos::TeamPolicy<>::member_type& team) const;

  struct TagCopyLocalXShmem {};
  KOKKOS_INLINE_FUNCTION
  void operator() (TagCopyLocalXShmem, const T_INT& i) const;
};

#define FORCE_MODULES_EXTERNAL_TEMPLATE
#define FORCETYPE_DECLARE_TEMPLATE_MACRO(NeighType) ForceSNAP<NeighType>
#include<modules_neighbor.h>
#undef FORCETYPE_DECLARE_TEMPLATE_MACRO
#undef FORCE_MODULES_EXTERNAL_TEMPLATE

#endif
#endif
