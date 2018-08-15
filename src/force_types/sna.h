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

/* -*- c++ -*- -------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Aidan Thompson, Christian Trott, SNL
------------------------------------------------------------------------- */

#ifndef LMP_SNA_H
#define LMP_SNA_H

#include <complex>
#include <ctime>
#include <Kokkos_Core.hpp>
#include <types.h>

struct SNA_LOOPINDICES {
  int j1, j2, j;
};

class SNA {

public:
  typedef Kokkos::View<int*> t_sna_1i;
  typedef Kokkos::View<double*> t_sna_1d;
  typedef Kokkos::View<double**, Kokkos::LayoutRight> t_sna_2d;
  typedef Kokkos::View<double***, Kokkos::LayoutRight> t_sna_3d;
  typedef Kokkos::View<double***, Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Atomic> > t_sna_3d_atomic;
  typedef Kokkos::View<double***[3], Kokkos::LayoutRight> t_sna_4d;
  typedef Kokkos::View<double**[3], Kokkos::LayoutRight> t_sna_3d3;
  typedef Kokkos::View<double*****, Kokkos::LayoutRight> t_sna_5d;
  inline
  SNA() {};
  KOKKOS_INLINE_FUNCTION
  SNA(const SNA& sna, const Kokkos::TeamPolicy<>::member_type& team);
  inline
  SNA(double, int, int, int, double, int, int);

  KOKKOS_INLINE_FUNCTION
  ~SNA();
  inline
  void build_indexlist(); // SNA()
  inline
  void init();            //
  inline
  T_INT size_team_scratch_arrays();
  inline
  T_INT size_thread_scratch_arrays();

  int ncoeff;

  // functions for bispectrum coefficients

  KOKKOS_INLINE_FUNCTION
  void compute_ui(const Kokkos::TeamPolicy<>::member_type& team, int); // ForceSNAP
  KOKKOS_INLINE_FUNCTION
  void compute_zi(const Kokkos::TeamPolicy<>::member_type& team);    // ForceSNAP

  // functions for derivatives

  KOKKOS_INLINE_FUNCTION
  void compute_duidrj(const Kokkos::TeamPolicy<>::member_type& team, double*, double, double); //ForceSNAP
  KOKKOS_INLINE_FUNCTION
  void compute_dbidrj(const Kokkos::TeamPolicy<>::member_type& team); //ForceSNAP
  KOKKOS_INLINE_FUNCTION
  void copy_dbi2dbvec(const Kokkos::TeamPolicy<>::member_type& team); //ForceSNAP
  KOKKOS_INLINE_FUNCTION
  double compute_sfac(double, double); // add_uarraytot, compute_duarray
  KOKKOS_INLINE_FUNCTION
  double compute_dsfac(double, double); // compute_duarray

#ifdef TIMING_INFO
  double* timers;
  timespec starttime, endtime;
  int print;
  int counter;
#endif

  //per sna class instance for OMP use


  // Per InFlight Particle
  t_sna_2d rij;
  t_sna_1i inside;
  t_sna_1d wj;
  t_sna_1d rcutij;
  int nmax;

  void grow_rij(int);

  int twojmax, diagonalstyle;
  // Per InFlight Particle
  t_sna_3d uarraytot_r, uarraytot_i;
  t_sna_3d_atomic uarraytot_r_a, uarraytot_i_a;
  t_sna_5d zarray_r, zarray_i;

  // Per InFlight Interaction
  t_sna_3d uarray_r, uarray_i;

  // derivatives of data
  Kokkos::View<double*[3], Kokkos::LayoutRight> dbvec;
  t_sna_4d duarray_r, duarray_i;
  t_sna_4d dbarray;

private:
  double rmin0, rfac0;

  //use indexlist instead of loops, constructor generates these
  // Same accross all SNA
  Kokkos::View<SNA_LOOPINDICES*> idxj,idxj_full;
  int idxj_max,idxj_full_max;
  // data for bispectrum coefficients

  // Same accross all SNA
  t_sna_5d cgarray;
  t_sna_2d rootpqarray;


  static const int nmaxfactorial = 167;
  KOKKOS_INLINE_FUNCTION
  double factorial(int);

  KOKKOS_INLINE_FUNCTION
  void create_team_scratch_arrays(const Kokkos::TeamPolicy<>::member_type& team); // SNA()
  KOKKOS_INLINE_FUNCTION
  void create_thread_scratch_arrays(const Kokkos::TeamPolicy<>::member_type& team); // SNA()
  inline
  void init_clebsch_gordan(); // init()
  inline
  void init_rootpqarray();    // init()
  KOKKOS_INLINE_FUNCTION
  void zero_uarraytot(const Kokkos::TeamPolicy<>::member_type& team);      // compute_ui
  KOKKOS_INLINE_FUNCTION
  void addself_uarraytot(const Kokkos::TeamPolicy<>::member_type& team, double); // compute_ui
  KOKKOS_INLINE_FUNCTION
  void add_uarraytot(const Kokkos::TeamPolicy<>::member_type& team, double, double, double); // compute_ui

  KOKKOS_INLINE_FUNCTION
  void compute_uarray(const Kokkos::TeamPolicy<>::member_type& team,
                      double, double, double,
                      double, double); // compute_ui
  KOKKOS_INLINE_FUNCTION
  double deltacg(int, int, int);  // init_clebsch_gordan
  inline
  int compute_ncoeff();           // SNA()
  KOKKOS_INLINE_FUNCTION
  void compute_duarray(const Kokkos::TeamPolicy<>::member_type& team,
                       double, double, double, // compute_duidrj
                       double, double, double, double, double);

  // if number of atoms are small use per atom arrays
  // for twojmax arrays, rij, inside, bvec
  // this will increase the memory footprint considerably,
  // but allows parallel filling and reuse of these arrays
  int use_shared_arrays;

  // Sets the style for the switching function
  // 0 = none
  // 1 = cosine
  int switch_flag;

  // Self-weight
  double wself;
};

#include<sna_impl.hpp>
#endif

/* ERROR/WARNING messages:

E: Invalid argument to factorial %d

N must be >= 0 and <= 167, otherwise the factorial result is too
large.

*/
