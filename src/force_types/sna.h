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

#ifdef __SYCL_DEVICE_ONLY__
#include <CL/sycl.hpp>
#endif

struct WignerWrapper {
  using double = double;
  using complex = SNAComplex<double>;
  static constexpr int vector_length = vector_length_;

  const int offset; // my offset into the vector (0, ..., vector_length - 1)
  double* buffer; // buffer of real numbers

  KOKKOS_INLINE_FUNCTION
  WignerWrapper(complex* buffer_, const int offset_)
   : offset(offset_), buffer(reinterpret_cast<double*>(buffer_))
  { ; }

  KOKKOS_INLINE_FUNCTION
  complex get(const int& ma) const {
    return complex(buffer[offset + 2 * vector_length * ma], buffer[offset + vector_length + 2 * vector_length * ma]);
  }

  KOKKOS_INLINE_FUNCTION
  void set(const int& ma, const complex& store) const {
    buffer[offset + 2 * vector_length * ma] = store.re;
    buffer[offset + vector_length + 2 * vector_length * ma] = store.im;
  }
};

struct alignas(8) FullHalfMapper {
  int idxu_half;
  int flip_sign; // 0 -> isn't flipped, 1 -> conj, -1 -> -conj
};

template<class DeviceType, typename double_, int vector_length_>
class SNA {

 public:
  using double = double_;
  using complex = SNAComplex<double>;
  static constexpr int vector_length = vector_length_;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  typedef Kokkos::View<int*, DeviceType> t_sna_1i;
  typedef Kokkos::View<double*, DeviceType> t_sna_1d;
  typedef Kokkos::View<double*, KKDeviceType, Kokkos::MemoryTraits<Kokkos::Atomic>> t_sna_1d_atomic;
  typedef Kokkos::View<int**, DeviceType> t_sna_2i;
  typedef Kokkos::View<double**, DeviceType> t_sna_2d;
  typedef Kokkos::View<double**, Kokkos::LayoutLeft, DeviceType> t_sna_2d_ll;
  typedef Kokkos::View<double***, DeviceType> t_sna_3d;
  typedef Kokkos::View<double***, Kokkos::LayoutLeft, DeviceType> t_sna_3d_ll;
  typedef Kokkos::View<double***[3], DeviceType> t_sna_4d;
  typedef Kokkos::View<double****, Kokkos::LayoutLeft, DeviceType> t_sna_4d_ll;
  typedef Kokkos::View<double**[3], DeviceType> t_sna_3d3;
  typedef Kokkos::View<double*****, DeviceType> t_sna_5d;

  typedef Kokkos::View<complex*, DeviceType> t_sna_1c;
  typedef Kokkos::View<complex*, KKDeviceType, Kokkos::MemoryTraits<Kokkos::Atomic>> t_sna_1c_atomic;
  typedef Kokkos::View<complex**, DeviceType> t_sna_2c;
  typedef Kokkos::View<complex**, Kokkos::LayoutLeft, DeviceType> t_sna_2c_ll;
  typedef Kokkos::View<complex**, Kokkos::LayoutRight, DeviceType> t_sna_2c_lr;
  typedef Kokkos::View<complex***, DeviceType> t_sna_3c;
  typedef Kokkos::View<complex***, Kokkos::LayoutLeft, DeviceType> t_sna_3c_ll;
  typedef Kokkos::View<complex***[3], DeviceType> t_sna_4c;
  typedef Kokkos::View<complex***[3], Kokkos::LayoutLeft, DeviceType> t_sna_4c3_ll;
  typedef Kokkos::View<complex****, Kokkos::LayoutLeft, DeviceType> t_sna_4c_ll;
  typedef Kokkos::View<complex**[3], DeviceType> t_sna_3c3;
  typedef Kokkos::View<complex*****, DeviceType> t_sna_5c;

  inline
  SNA() {};

  KOKKOS_INLINE_FUNCTION
  SNA(const SNA<DeviceType,double,vector_length>& sna, const typename Kokkos::TeamPolicy<DeviceType>::member_type& team);

  inline
  SNA(double, int, double, int, int, int, int, int, int, int);

  KOKKOS_INLINE_FUNCTION
  ~SNA();

  inline
  void build_indexlist(); // SNA()

  inline
  void init();            //

  double memory_usage();

  int ncoeff;
  int host_flag;

  // functions for bispectrum coefficients, GPU only
  KOKKOS_INLINE_FUNCTION
  void compute_cayley_klein(const int&, const int&, const int&);
  KOKKOS_INLINE_FUNCTION
  void pre_ui(const int&, const int&, const int&, const int&); // ForceSNAP

  // version of the code with parallelism over j_bend
  KOKKOS_INLINE_FUNCTION
  void compute_ui_small(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, const int, const int, const int, const int); // ForceSNAP
  // version of the code without parallelism over j_bend
  KOKKOS_INLINE_FUNCTION
  void compute_ui_large(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, const int, const int, const int); // ForceSNAP

  KOKKOS_INLINE_FUNCTION
  void compute_zi(const int&, const int&, const int&);    // ForceSNAP
  KOKKOS_INLINE_FUNCTION
  void compute_yi(int,int,int,
   const Kokkos::View<double***, Kokkos::LayoutLeft, DeviceType> &beta_pack); // ForceSNAP
  KOKKOS_INLINE_FUNCTION
  void compute_yi_with_zlist(int,int,int,
   const Kokkos::View<double***, Kokkos::LayoutLeft, DeviceType> &beta_pack); // ForceSNAP
  KOKKOS_INLINE_FUNCTION
  void compute_bi(const int&, const int&, const int&);    // ForceSNAP

  // functions for derivatives, GPU only
  // version of the code with parallelism over j_bend
  template<int dir>
  KOKKOS_INLINE_FUNCTION
  void compute_fused_deidrj_small(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, const int, const int, const int, const int); //ForceSNAP
  // version of the code without parallelism over j_bend
  template<int dir>
  KOKKOS_INLINE_FUNCTION
  void compute_fused_deidrj_large(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, const int, const int, const int); //ForceSNAP

  // core "evaluation" functions that get plugged into "compute" functions
  // plugged into compute_ui_small, compute_ui_large
  KOKKOS_FORCEINLINE_FUNCTION
  void evaluate_ui_jbend(const WignerWrapper<double, vector_length>&, const complex&, const complex&, const double&, const int&,
                        const int&, const int&, const int&);
  // plugged into compute_zi, compute_yi
  KOKKOS_FORCEINLINE_FUNCTION
  complex evaluate_zi(const int&, const int&, const int&, const int&, const int&, const int&, const int&, const int&, const int&,
                        const int&, const int&, const int&, const int&, const double*);
  // plugged into compute_yi, compute_yi_with_zlist
  KOKKOS_FORCEINLINE_FUNCTION
  double evaluate_beta_scaled(const int&, const int&, const int&, const int&, const int&, const int&, const int&, const int&,
                        const Kokkos::View<double***, Kokkos::LayoutLeft, DeviceType> &);
  // plugged into compute_fused_deidrj_small, compute_fused_deidrj_large
  KOKKOS_FORCEINLINE_FUNCTION
  double evaluate_duidrj_jbend(const WignerWrapper<double, vector_length>&, const complex&, const complex&, const double&,
                        const WignerWrapper<double, vector_length>&, const complex&, const complex&, const double&,
                        const int&, const int&, const int&, const int&);

  // functions for bispectrum coefficients, CPU only
  KOKKOS_INLINE_FUNCTION
  void pre_ui_cpu(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team,const int&,const int&); // ForceSNAP
  KOKKOS_INLINE_FUNCTION
  void compute_ui_cpu(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, int, int); // ForceSNAP
  KOKKOS_INLINE_FUNCTION
  void compute_zi_cpu(const int&);    // ForceSNAP
  KOKKOS_INLINE_FUNCTION
  void compute_yi_cpu(int,
   const Kokkos::View<double**, DeviceType> &beta); // ForceSNAP
    KOKKOS_INLINE_FUNCTION
  void compute_bi_cpu(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, int);    // ForceSNAP

  // functions for derivatives, CPU only
  KOKKOS_INLINE_FUNCTION
  void compute_duidrj_cpu(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, int, int); //ForceSNAP
  KOKKOS_INLINE_FUNCTION
  void compute_deidrj_cpu(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, int, int); // ForceSNAP

  KOKKOS_INLINE_FUNCTION
  double compute_sfac(double, double, double, double); // add_uarraytot, compute_duarray

  KOKKOS_INLINE_FUNCTION
  double compute_dsfac(double, double, double, double); // compute_duarray

  KOKKOS_INLINE_FUNCTION
  void compute_s_dsfac(const double, const double, const double, const double, double&, double&); // compute_cayley_klein

#ifdef TIMING_INFO
  double* timers;
  timespec starttime, endtime;
  int print;
  int counter;
#endif

  //per sna class instance for OMP use

  // Per InFlight Particle
  t_sna_3d rij;
  t_sna_2i inside;
  t_sna_2d wj;
  t_sna_2d rcutij;
  t_sna_2d sinnerij;
  t_sna_2d dinnerij;
  t_sna_2i element;
  t_sna_3d dedr;
  int natom, nmax;

  void grow_rij(int, int);

  int twojmax, diagonalstyle;

  t_sna_3d blist;
  t_sna_3c_ll ulisttot;
  t_sna_3c_ll ulisttot_full; // un-folded ulisttot, cpu only
  t_sna_3c_ll zlist;

  t_sna_3c_ll ulist;
  t_sna_3c_ll ylist;

  // derivatives of data
  t_sna_4c3_ll dulist;

  // Modified structures for GPU backend
  t_sna_3c_ll a_pack; // Cayley-Klein `a`
  t_sna_3c_ll b_pack; // `b`
  t_sna_4c_ll da_pack; // `da`
  t_sna_4c_ll db_pack; // `db`
  t_sna_4d_ll sfac_pack; // sfac, dsfac_{x,y,z}

  t_sna_4d_ll ulisttot_re_pack; // split real,
  t_sna_4d_ll ulisttot_im_pack; // imag, AoSoA, flattened
  t_sna_4c_ll ulisttot_pack; // AoSoA layout
  t_sna_4c_ll zlist_pack; // AoSoA layout
  t_sna_4d_ll blist_pack;
  t_sna_4d_ll ylist_pack_re; // split real,
  t_sna_4d_ll ylist_pack_im; // imag AoSoA layout

  int idxcg_max, idxu_max, idxu_half_max, idxu_cache_max, idxz_max, idxb_max;

  // Chem snap counts
  int nelements;
  int ndoubles;
  int ntriples;

 private:
  double rmin0, rfac0;

  //use indexlist instead of loops, constructor generates these
  // Same across all SNA
  Kokkos::View<int*[10], DeviceType> idxz;
  Kokkos::View<int*[3], DeviceType> idxb;
  Kokkos::View<int***, DeviceType> idxcg_block;

 public:
  Kokkos::View<int*, DeviceType> idxu_block;
  Kokkos::View<int*, DeviceType> idxu_half_block;
  Kokkos::View<int*, DeviceType> idxu_cache_block;
  Kokkos::View<FullHalfMapper*, DeviceType> idxu_full_half;

 private:
  Kokkos::View<int***, DeviceType> idxz_block;
  Kokkos::View<int***, DeviceType> idxb_block;

  // data for bispectrum coefficients

  // Same across all SNA
  t_sna_1d cglist;
  t_sna_2d rootpqarray;

  static const int nmaxfactorial = 167;
  static const double nfac_table[];
  inline
  double factorial(int);

  KOKKOS_INLINE_FUNCTION
  void create_team_scratch_arrays(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team); // SNA()
  KOKKOS_INLINE_FUNCTION
  void create_thread_scratch_arrays(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team); // SNA()

  inline
  void init_clebsch_gordan(); // init()

  inline
  void init_rootpqarray();    // init()

  KOKKOS_INLINE_FUNCTION
  void add_uarraytot(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, int, int, const double&, const double&, const double&, const double&, const double&, int); // compute_ui

  KOKKOS_INLINE_FUNCTION
  void compute_uarray_cpu(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, int, int,
                      const double&, const double&, const double&,
                      const double&, const double&); // compute_ui_cpu


  inline
  double deltacg(int, int, int);  // init_clebsch_gordan

  inline
  int compute_ncoeff();           // SNA()
  KOKKOS_INLINE_FUNCTION
  void compute_duarray_cpu(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, int, int,
                       const double&, const double&, const double&, // compute_duidrj_cpu
                           const double&, const double&, const double&, const double&, const double&,
                           const double&, const double&);

  // Sets the style for the switching function
  // 0 = none
  // 1 = cosine
  int switch_flag;

  // Sets the style for the inner switching function
  // 0 = none
  // 1 = cosine
  int switch_inner_flag;

  // Chem snap flags
  int chem_flag;
  int bnorm_flag;

  // Self-weight
  double wself;
  int wselfall_flag;

  int bzero_flag; // 1 if bzero subtracted from barray
  Kokkos::View<double*, DeviceType> bzero; // array of B values for isolated atoms
};

#include<sna_impl.hpp>
#endif

