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

struct SNA_LOOPINDICES {
  int j1, j2, j;
};

class SNA {

public:
  typedef Kokkos::View<int*> t_sna_1i;
  typedef Kokkos::View<double*> t_sna_1d;
  typedef Kokkos::View<double**> t_sna_2d;
  typedef Kokkos::View<double***> t_sna_3d;
  typedef Kokkos::View<double***[3]> t_sna_4d;
  typedef Kokkos::View<double**[3]> t_sna_3d3;
  typedef Kokkos::View<double*****> t_sna_5d;
  SNA(double, int, int, int, double, int, int);

  ~SNA();
  void build_indexlist();
  void init();
  double memory_usage();

  int ncoeff;

  // functions for bispectrum coefficients

  void compute_ui(int);
  void compute_ui_omp(int, int);
  void compute_zi();
  void compute_zi_omp(int);
  void compute_bi();
  void copy_bi2bvec();

  // functions for derivatives

  void compute_duidrj(double*, double, double);
  void compute_dbidrj();
  void compute_dbidrj_nonsymm();
  void copy_dbi2dbvec();
  double compute_sfac(double, double);
  double compute_dsfac(double, double);

#ifdef TIMING_INFO
  double* timers;
  timespec starttime, endtime;
  int print;
  int counter;
#endif

  //per sna class instance for OMP use

  //double* bvec, ** dbvec;
  Kokkos::View<double*> bvec;
  Kokkos::View<double*[3]> dbvec;

  t_sna_2d rij;
  t_sna_1i inside;
  t_sna_1d wj;
  t_sna_1d rcutij;
  int nmax;

  void grow_rij(int);

  int twojmax, diagonalstyle;
  t_sna_3d uarraytot_r, uarraytot_i;
  t_sna_5d zarray_r, zarray_i;
  t_sna_3d uarraytot_r_b, uarraytot_i_b;
  t_sna_5d zarray_r_b, zarray_i_b;
  t_sna_3d uarray_r, uarray_i;

private:
  double rmin0, rfac0;

  //use indexlist instead of loops, constructor generates these
  SNA_LOOPINDICES* idxj;
  int idxj_max;
  // data for bispectrum coefficients

  t_sna_5d cgarray;
  t_sna_2d rootpqarray;
  t_sna_3d barray;

  // derivatives of data

  t_sna_4d duarray_r, duarray_i;
  t_sna_4d dbarray;

  static const int nmaxfactorial = 167;
  static const double nfac_table[];
  double factorial(int);

  void create_twojmax_arrays();
  void destroy_twojmax_arrays();
  void init_clebsch_gordan();
  void init_rootpqarray();
  void jtostr(char*, int);
  void mtostr(char*, int, int);
  void print_clebsch_gordan(FILE*);
  void zero_uarraytot();
  void addself_uarraytot(double);
  void add_uarraytot(double, double, double);
  void add_uarraytot_omp(double, double, double);
  void compute_uarray(double, double, double,
                      double, double);
  void compute_uarray_omp(double, double, double,
                          double, double, int);
  double deltacg(int, int, int);
  int compute_ncoeff();
  void compute_duarray(double, double, double,
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

  int bzero_flag; // 1 if bzero subtracted from barray
  double *bzero;  // array of B values for isolated atoms
};


#endif

/* ERROR/WARNING messages:

E: Invalid argument to factorial %d

N must be >= 0 and <= 167, otherwise the factorial result is too
large.

*/
