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
  void build_indexlist(); // SNA()
  void init();            //

  int ncoeff;

  // functions for bispectrum coefficients

  void compute_ui(int); // ForceSNAP
  void compute_zi();    // ForceSNAP

  // functions for derivatives

  void compute_duidrj(double*, double, double); //ForceSNAP
  void compute_dbidrj(); //ForceSNAP
  void copy_dbi2dbvec(); //ForceSNAP
  double compute_sfac(double, double); // add_uarraytot, compute_duarray
  double compute_dsfac(double, double); // compute_duarray

#ifdef TIMING_INFO
  double* timers;
  timespec starttime, endtime;
  int print;
  int counter;
#endif

  //per sna class instance for OMP use

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
  t_sna_3d uarray_r, uarray_i;

private:
  double rmin0, rfac0;

  //use indexlist instead of loops, constructor generates these
  SNA_LOOPINDICES* idxj;
  int idxj_max;
  // data for bispectrum coefficients

  t_sna_5d cgarray;
  t_sna_2d rootpqarray;

  // derivatives of data

  t_sna_4d duarray_r, duarray_i;
  t_sna_4d dbarray;

  static const int nmaxfactorial = 167;
  double factorial(int);

  void create_twojmax_arrays(); // SNA()
  void destroy_twojmax_arrays(); // ~SNA()
  void init_clebsch_gordan(); // init()
  void init_rootpqarray();    // init()
  void zero_uarraytot();      // compute_ui
  void addself_uarraytot(double); // compute_ui
  void add_uarraytot(double, double, double); // compute_ui

  void compute_uarray(double, double, double,
                      double, double); // compute_ui
  double deltacg(int, int, int);  // init_clebsch_gordan
  int compute_ncoeff();           // SNA()
  void compute_duarray(double, double, double, // compute_duidrj
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


#endif

/* ERROR/WARNING messages:

E: Invalid argument to factorial %d

N must be >= 0 and <= 167, otherwise the factorial result is too
large.

*/
