/* ----------------------------------------------------------------------
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

#include "sna.h"
#include <math.h>
#include "math_const.h"
//#include "math_extra.h"
#include <string.h>
#include <stdlib.h>

using namespace std;
using namespace MathConst;

/* ----------------------------------------------------------------------

   this implementation is based on the method outlined
   in Bartok[1], using formulae from VMK[2].

   for the Clebsch-Gordan coefficients, we
   convert the VMK half-integral labels
   a, b, c, alpha, beta, gamma
   to array offsets j1, j2, j, m1, m2, m
   using the following relations:

   j1 = 2*a
   j2 = 2*b
   j =  2*c

   m1 = alpha+a      2*alpha = 2*m1 - j1
   m2 = beta+b    or 2*beta = 2*m2 - j2
   m =  gamma+c      2*gamma = 2*m - j

   in this way:

   -a <= alpha <= a
   -b <= beta <= b
   -c <= gamma <= c

   becomes:

   0 <= m1 <= j1
   0 <= m2 <= j2
   0 <= m <= j

   and the requirement that
   a+b+c be integral implies that
   j1+j2+j must be even.
   The requirement that:

   gamma = alpha+beta

   becomes:

   2*m - j = 2*m1 - j1 + 2*m2 - j2

   Similarly, for the Wigner U-functions U(J,m,m') we
   convert the half-integral labels J,m,m' to
   array offsets j,ma,mb:

   j = 2*J
   ma = J+m
   mb = J+m'

   so that:

   0 <= j <= 2*Jmax
   0 <= ma, mb <= j.

   For the bispectrum components B(J1,J2,J) we convert to:

   j1 = 2*J1
   j2 = 2*J2
   j = 2*J

   and the requirement:

   |J1-J2| <= J <= J1+J2, for j1+j2+j integral

   becomes:

   |j1-j2| <= j <= j1+j2, for j1+j2+j even integer

   or

   j = |j1-j2|, |j1-j2|+2,...,j1+j2-2,j1+j2

   [1] Albert Bartok-Partay, "Gaussian Approximation..."
   Doctoral Thesis, Cambrindge University, (2009)

   [2] D. A. Varshalovich, A. N. Moskalev, and V. K. Khersonskii,
   "Quantum Theory of Angular Momentum," World Scientific (1988)

------------------------------------------------------------------------- */

SNA::SNA(double rfac0_in,
         int twojmax_in, int diagonalstyle_in, int use_shared_arrays_in,
         double rmin0_in, int switch_flag_in, int bzero_flag_in) 
{
  wself = 1.0;
  
  use_shared_arrays = use_shared_arrays_in;
  rfac0 = rfac0_in;
  rmin0 = rmin0_in;
  switch_flag = switch_flag_in;
  bzero_flag = bzero_flag_in;

  twojmax = twojmax_in;
  diagonalstyle = diagonalstyle_in;

  ncoeff = compute_ncoeff();

  create_twojmax_arrays();

  //printf("SNAP-COMPARE: NCOEFF: %i\n",ncoeff);
  bvec = Kokkos::View<double*>("pair:bvec",ncoeff);
  dbvec = Kokkos::View<double*[3]>("pair:dbvec",ncoeff);
  nmax = 0;
  idxj = NULL;

  if (bzero_flag) {
    double www = wself*wself*wself;
    for(int j = 0; j <= twojmax; j++)
      bzero[j] = www*(j+1);
  }
  
#ifdef TIMING_INFO
  timers = new double[20];
  for(int i = 0; i < 20; i++) timers[i] = 0;
  print = 0;
  counter = 0;
#endif

  build_indexlist();

  
}

/* ---------------------------------------------------------------------- */

SNA::~SNA()
{
  if(!use_shared_arrays) {
    destroy_twojmax_arrays();
/*    memory->destroy(rij);
    memory->destroy(inside);
    memory->destroy(wj);
    memory->destroy(rcutij);
    memory->destroy(bvec);
    memory->destroy(dbvec);*/
  }
  delete[] idxj;
}

void SNA::build_indexlist()
{
  if(diagonalstyle == 3) {
    int idxj_count = 0;

    for(int j1 = 0; j1 <= twojmax; j1++)
      for(int j2 = 0; j2 <= j1; j2++)
        for(int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
          if (j >= j1) idxj_count++;

    // indexList can be changed here

    //printf("SNAP-COMPARE C: %i\n",idxj_count);
    idxj = new SNA_LOOPINDICES[idxj_count];
    idxj_max = idxj_count;

    idxj_count = 0;

    for(int j1 = 0; j1 <= twojmax; j1++)
      for(int j2 = 0; j2 <= j1; j2++)
        for(int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
	  if (j >= j1) {
	    idxj[idxj_count].j1 = j1;
	    idxj[idxj_count].j2 = j2;
	    idxj[idxj_count].j = j;
	    idxj_count++;
	  }
  }

}
/* ---------------------------------------------------------------------- */

void SNA::init()
{
  init_clebsch_gordan();
  init_rootpqarray();
}


void SNA::grow_rij(int newnmax)
{
  if(newnmax <= nmax) return;

  nmax = newnmax;

  if(!use_shared_arrays) {
    rij = t_sna_2d("SNA::rij",nmax,3);
    rcutij = t_sna_1d("SNA::rcutij",nmax);
    wj = t_sna_1d("SNA::wj",nmax);
    inside = t_sna_1i("SNA::inside",nmax);
    /*memory->destroy(rij);
    memory->destroy(inside);
    memory->destroy(wj);
    memory->destroy(rcutij);
    memory->create(rij, nmax, 3, "pair:rij");
    memory->create(inside, nmax, "pair:inside");
    memory->create(wj, nmax, "pair:wj");
    memory->create(rcutij, nmax, "pair:rcutij");*/
 }
}
/* ----------------------------------------------------------------------
   compute Ui by summing over neighbors j
------------------------------------------------------------------------- */

void SNA::compute_ui(int jnum)
{
  double rsq, r, x, y, z, z0, theta0;

  // utot(j,ma,mb) = 0 for all j,ma,ma
  // utot(j,ma,ma) = 1 for all j,ma
  // for j in neighbors of i:
  //   compute r0 = (x,y,z,z0)
  //   utot(j,ma,mb) += u(r0;j,ma,mb) for all j,ma,mb

  zero_uarraytot();
  addself_uarraytot(wself);

#ifdef TIMING_INFO
  clock_gettime(CLOCK_REALTIME, &starttime);
#endif

  for(int j = 0; j < jnum; j++) {
    x = rij(j,0);
    y = rij(j,1);
    z = rij(j,2);
    rsq = x * x + y * y + z * z;
    r = sqrt(rsq);

    theta0 = (r - rmin0) * rfac0 * MY_PI / (rcutij[j] - rmin0);
    //    theta0 = (r - rmin0) * rscale0;
    z0 = r / tan(theta0);

    compute_uarray(x, y, z, z0, r);
    add_uarraytot(r, wj[j], rcutij[j]);
  }

#ifdef TIMING_INFO
  clock_gettime(CLOCK_REALTIME, &endtime);
  timers[0] += (endtime.tv_sec - starttime.tv_sec + 1.0 *
                (endtime.tv_nsec - starttime.tv_nsec) / 1000000000);
#endif

}

/* ----------------------------------------------------------------------
   compute Zi by summing over products of Ui
------------------------------------------------------------------------- */

void SNA::compute_zi()
{
  // for j1 = 0,...,twojmax
  //   for j2 = 0,twojmax
  //     for j = |j1-j2|,Min(twojmax,j1+j2),2
  //        for ma = 0,...,j
  //          for mb = 0,...,jmid
  //            z(j1,j2,j,ma,mb) = 0
  //            for ma1 = Max(0,ma+(j1-j2-j)/2),Min(j1,ma+(j1+j2-j)/2)
  //              sumb1 = 0
  //              ma2 = ma-ma1+(j1+j2-j)/2;
  //              for mb1 = Max(0,mb+(j1-j2-j)/2),Min(j1,mb+(j1+j2-j)/2)
  //                mb2 = mb-mb1+(j1+j2-j)/2;
  //                sumb1 += cg(j1,mb1,j2,mb2,j) *
  //                  u(j1,ma1,mb1) * u(j2,ma2,mb2)
  //              z(j1,j2,j,ma,mb) += sumb1*cg(j1,ma1,j2,ma2,j)

#ifdef TIMING_INFO
  clock_gettime(CLOCK_REALTIME, &starttime);
#endif

  // compute_dbidrj() requires full j1/j2/j chunk of z elements
  // use zarray j1/j2 symmetry

  for(int j1 = 0; j1 <= twojmax; j1++)
    for(int j2 = 0; j2 <= j1; j2++) {
      for(int j = j1 - j2; j <= MIN(twojmax, j1 + j2); j += 2) {
	double sumb1_r, sumb1_i;
	int ma2, mb2;
	for(int mb = 0; 2*mb <= j; mb++)
	  for(int ma = 0; ma <= j; ma++) {
	    zarray_r(j1,j2,j,ma,mb) = 0.0;
	    zarray_i(j1,j2,j,ma,mb) = 0.0;

	    for(int ma1 = MAX(0, (2 * ma - j - j2 + j1) / 2);
		ma1 <= MIN(j1, (2 * ma - j + j2 + j1) / 2); ma1++) {
	      sumb1_r = 0.0;
	      sumb1_i = 0.0;

	      ma2 = (2 * ma - j - (2 * ma1 - j1) + j2) / 2;

	      for(int mb1 = MAX(0, (2 * mb - j - j2 + j1) / 2);
              mb1 <= MIN(j1, (2 * mb - j + j2 + j1) / 2); mb1++) {

		mb2 = (2 * mb - j - (2 * mb1 - j1) + j2) / 2;
		sumb1_r += cgarray(j1,j2,j,mb1,mb2) *
		  (uarraytot_r(j1,ma1,mb1) * uarraytot_r(j2,ma2,mb2) -
		   uarraytot_i(j1,ma1,mb1) * uarraytot_i(j2,ma2,mb2));
		sumb1_i += cgarray(j1,j2,j,mb1,mb2) *
		  (uarraytot_r(j1,ma1,mb1) * uarraytot_i(j2,ma2,mb2) +
		   uarraytot_i(j1,ma1,mb1) * uarraytot_r(j2,ma2,mb2));
	      } // end loop over mb1

	      zarray_r(j1,j2,j,ma,mb) +=
		sumb1_r * cgarray(j1,j2,j,ma1,ma2);
	      zarray_i(j1,j2,j,ma,mb) +=
		sumb1_i * cgarray(j1,j2,j,ma1,ma2);
	    } // end loop over ma1
	  } // end loop over ma, mb
      } // end loop over j
    } // end loop over j1, j2

#ifdef TIMING_INFO
  clock_gettime(CLOCK_REALTIME, &endtime);
  timers[1] += (endtime.tv_sec - starttime.tv_sec + 1.0 *
                (endtime.tv_nsec - starttime.tv_nsec) / 1000000000);
#endif
}


/* ----------------------------------------------------------------------
   calculate derivative of Ui w.r.t. atom j
------------------------------------------------------------------------- */

void SNA::compute_duidrj(double* rij, double wj, double rcut)
{
  double rsq, r, x, y, z, z0, theta0, cs, sn;
  double dz0dr;

  x = rij[0];
  y = rij[1];
  z = rij[2];
  rsq = x * x + y * y + z * z;
  r = sqrt(rsq);
  double rscale0 = rfac0 * MY_PI / (rcut - rmin0);
  theta0 = (r - rmin0) * rscale0;
  cs = cos(theta0);
  sn = sin(theta0);
  z0 = r * cs / sn;
  dz0dr = z0 / r - (r*rscale0) * (rsq + z0 * z0) / rsq;

#ifdef TIMING_INFO
  clock_gettime(CLOCK_REALTIME, &starttime);
#endif

  compute_duarray(x, y, z, z0, r, dz0dr, wj, rcut);

#ifdef TIMING_INFO
  clock_gettime(CLOCK_REALTIME, &endtime);
  timers[3] += (endtime.tv_sec - starttime.tv_sec + 1.0 *
                (endtime.tv_nsec - starttime.tv_nsec) / 1000000000);
#endif

}

/* ----------------------------------------------------------------------
   calculate derivative of Bi w.r.t. atom j
   variant using indexlist for j1,j2,j
   variant using symmetry relation
------------------------------------------------------------------------- */

void SNA::compute_dbidrj()
{
  // for j1 = 0,...,twojmax
  //   for j2 = 0,twojmax
  //     for j = |j1-j2|,Min(twojmax,j1+j2),2
  //        zdb = 0
  //        for mb = 0,...,jmid
  //          for ma = 0,...,j
  //            zdb +=
  //              Conj(dudr(j,ma,mb))*z(j1,j2,j,ma,mb)
  //        dbdr(j1,j2,j) += 2*zdb
  //        zdb = 0
  //        for mb1 = 0,...,j1mid
  //          for ma1 = 0,...,j1
  //            zdb +=
  //              Conj(dudr(j1,ma1,mb1))*z(j,j2,j1,ma1,mb1)
  //        dbdr(j1,j2,j) += 2*zdb*(j+1)/(j1+1)
  //        zdb = 0
  //        for mb2 = 0,...,j2mid
  //          for ma2 = 0,...,j2
  //            zdb +=
  //              Conj(dudr(j2,ma2,mb2))*z(j1,j,j2,ma2,mb2)
  //        dbdr(j1,j2,j) += 2*zdb*(j+1)/(j2+1)

  double* dbdr;
  double* dudr_r, *dudr_i;
  double sumzdu_r[3];
  double** jjjzarray_r;
  double** jjjzarray_i;
  double jjjmambzarray_r;
  double jjjmambzarray_i;

#ifdef TIMING_INFO
  clock_gettime(CLOCK_REALTIME, &starttime);
#endif

  for(int JJ = 0; JJ < idxj_max; JJ++) {
    const int j1 = idxj[JJ].j1;
    const int j2 = idxj[JJ].j2;
    const int j = idxj[JJ].j;

    dbdr = &dbarray(j1,j2,j,0);
    dbdr[0] = 0.0;
    dbdr[1] = 0.0;
    dbdr[2] = 0.0;

    // Sum terms Conj(dudr(j,ma,mb))*z(j1,j2,j,ma,mb)

    for(int k = 0; k < 3; k++)
      sumzdu_r[k] = 0.0;

    // use zarray j1/j2 symmetry (optional)

    int j_,j1_,j2_;
    if (j1 >= j2) {
      //jjjzarray_r = &zarray_r(j1,j2,j);
      //jjjzarray_i = &zarray_i(j1,j2,j);
      j1_ = j1;
      j2_ = j2;
      j_ = j;
    } else {
      j1_ = j2;
      j2_ = j1;
      j_ = j;
      //jjjzarray_r = &zarray_r(j2,j1,j);
      //jjjzarray_i = &zarray_i(j2,j1,j);
    }

    for(int mb = 0; 2*mb < j; mb++)
      for(int ma = 0; ma <= j; ma++) {

        dudr_r = &duarray_r(j,ma,mb,0);
        dudr_i = &duarray_i(j,ma,mb,0);
        jjjmambzarray_r = zarray_r(j1_,j2_,j_,ma,mb);
        jjjmambzarray_i = zarray_i(j1_,j2_,j_,ma,mb);
        for(int k = 0; k < 3; k++)
          sumzdu_r[k] +=
            dudr_r[k] * jjjmambzarray_r +
            dudr_i[k] * jjjmambzarray_i;

      } //end loop over ma mb

    // For j even, handle middle column

    if (j%2 == 0) {
      int mb = j/2;
      for(int ma = 0; ma < mb; ma++) {
        dudr_r = &duarray_r(j,ma,mb,0);
        dudr_i = &duarray_i(j,ma,mb,0);
        jjjmambzarray_r = zarray_r(j1_,j2_,j_,ma,mb);
        jjjmambzarray_i = zarray_i(j1_,j2_,j_,ma,mb);
        for(int k = 0; k < 3; k++)
          sumzdu_r[k] +=
            dudr_r[k] * jjjmambzarray_r +
        dudr_i[k] * jjjmambzarray_i;
      }
      int ma = mb;
      dudr_r = &duarray_r(j,ma,mb,0);
      dudr_i = &duarray_i(j,ma,mb,0);
      jjjmambzarray_r = zarray_r(j1_,j2_,j_,ma,mb);
      jjjmambzarray_i = zarray_i(j1_,j2_,j_,ma,mb);
      for(int k = 0; k < 3; k++)
        sumzdu_r[k] +=
            (dudr_r[k] * jjjmambzarray_r +
             dudr_i[k] * jjjmambzarray_i)*0.5;
    } // end if jeven

    for(int k = 0; k < 3; k++)
      dbdr[k] += 2.0*sumzdu_r[k];

    // Sum over Conj(dudr(j1,ma1,mb1))*z(j,j2,j1,ma1,mb1)

    double j1fac = (j+1)/(j1+1.0);

    for(int k = 0; k < 3; k++)
      sumzdu_r[k] = 0.0;

    // use zarray j1/j2 symmetry (optional)

    if (j >= j2) {
      j1_ = j;
      j2_ = j2;
      j_ = j1;

      //jjjzarray_r = zarray_r(j,j2,j1);
      //jjjzarray_i = zarray_i(j,j2,j1);
    } else {
      j1_ = j2;
      j2_ = j;
      j_ = j1;
      //jjjzarray_r = zarray_r(j2,j,j1);
      //jjjzarray_i = zarray_i(j2,j,j1);
    }

    for(int mb1 = 0; 2*mb1 < j1; mb1++)
      for(int ma1 = 0; ma1 <= j1; ma1++) {

        dudr_r = &duarray_r(j1,ma1,mb1,0);
        dudr_i = &duarray_i(j1,ma1,mb1,0);
        jjjmambzarray_r = zarray_r(j1_,j2_,j_,ma1,mb1);
        jjjmambzarray_i = zarray_i(j1_,j2_,j_,ma1,mb1);
        for(int k = 0; k < 3; k++)
          sumzdu_r[k] +=
            dudr_r[k] * jjjmambzarray_r +
            dudr_i[k] * jjjmambzarray_i;

      } //end loop over ma1 mb1

    // For j1 even, handle middle column

    if (j1%2 == 0) {
      int mb1 = j1/2;
      for(int ma1 = 0; ma1 < mb1; ma1++) {
        dudr_r = &duarray_r(j1,ma1,mb1,0);
        dudr_i = &duarray_i(j1,ma1,mb1,0);
        jjjmambzarray_r = zarray_r(j1_,j2_,j_,ma1,mb1);
	      jjjmambzarray_i = zarray_i(j1_,j2_,j_,ma1,mb1);
        for(int k = 0; k < 3; k++)
          sumzdu_r[k] +=
            dudr_r[k] * jjjmambzarray_r +
            dudr_i[k] * jjjmambzarray_i;
      }
      int ma1 = mb1;
      dudr_r = &duarray_r(j1,ma1,mb1,0);
      dudr_i = &duarray_i(j1,ma1,mb1,0);
      jjjmambzarray_r = zarray_r(j1_,j2_,j_,ma1,mb1);
      jjjmambzarray_i = zarray_i(j1_,j2_,j_,ma1,mb1);
      for(int k = 0; k < 3; k++)
        sumzdu_r[k] +=
            (dudr_r[k] * jjjmambzarray_r +
             dudr_i[k] * jjjmambzarray_i)*0.5;
    } // end if j1even

    for(int k = 0; k < 3; k++)
      dbdr[k] += 2.0*sumzdu_r[k]*j1fac;

    // Sum over Conj(dudr(j2,ma2,mb2))*z(j1,j,j2,ma2,mb2)

    double j2fac = (j+1)/(j2+1.0);

    for(int k = 0; k < 3; k++)
      sumzdu_r[k] = 0.0;

    // use zarray j1/j2 symmetry (optional)

    if (j1 >= j) {
      j1_ = j1;
      j2_ = j;
      j_ = j2;
      //jjjzarray_r = zarray_r(j1,j,j2);
      //jjjzarray_i = zarray_i(j1,j,j2);
    } else {
      j1_ = j;
      j2_ = j1;
      j_ = j2;
      //jjjzarray_r = zarray_r(j,j1,j2);
      //jjjzarray_i = zarray_i(j,j1,j2);
    }

    for(int mb2 = 0; 2*mb2 < j2; mb2++)
      for(int ma2 = 0; ma2 <= j2; ma2++) {

        dudr_r = &duarray_r(j2,ma2,mb2,0);
        dudr_i = &duarray_i(j2,ma2,mb2,0);
        jjjmambzarray_r = zarray_r(j1_,j2_,j_,ma2,mb2);
        jjjmambzarray_i = zarray_i(j1_,j2_,j_,ma2,mb2);
        for(int k = 0; k < 3; k++)
          sumzdu_r[k] +=
            dudr_r[k] * jjjmambzarray_r +
            dudr_i[k] * jjjmambzarray_i;
      } //end loop over ma2 mb2

    // For j2 even, handle middle column

    if (j2%2 == 0) {
      int mb2 = j2/2;
      for(int ma2 = 0; ma2 < mb2; ma2++) {
        dudr_r = &duarray_r(j2,ma2,mb2,0);
        dudr_i = &duarray_i(j2,ma2,mb2,0);
        jjjmambzarray_r = zarray_r(j1_,j2_,j_,ma2,mb2);
        jjjmambzarray_i = zarray_i(j1_,j2_,j_,ma2,mb2);
        for(int k = 0; k < 3; k++)
          sumzdu_r[k] +=
            dudr_r[k] * jjjmambzarray_r +
            dudr_i[k] * jjjmambzarray_i;
      }
      int ma2 = mb2;
      dudr_r = &duarray_r(j2,ma2,mb2,0);
      dudr_i = &duarray_i(j2,ma2,mb2,0);
      jjjmambzarray_r = zarray_r(j1_,j2_,j_,ma2,mb2);
      jjjmambzarray_i = zarray_i(j1_,j2_,j_,ma2,mb2);
      for(int k = 0; k < 3; k++)
        sumzdu_r[k] +=
          (dudr_r[k] * jjjmambzarray_r +
           dudr_i[k] * jjjmambzarray_i)*0.5;
    } // end if j2even

    for(int k = 0; k < 3; k++)
      dbdr[k] += 2.0*sumzdu_r[k]*j2fac;

  } //end loop over j1 j2 j

#ifdef TIMING_INFO
  clock_gettime(CLOCK_REALTIME, &endtime);
  timers[4] += (endtime.tv_sec - starttime.tv_sec + 1.0 *
                (endtime.tv_nsec - starttime.tv_nsec) / 1000000000);
#endif

}

/* ----------------------------------------------------------------------
   copy Bi derivatives into a vector
------------------------------------------------------------------------- */

void SNA::copy_dbi2dbvec()
{
  int ncount, j1, j2, j;

  ncount = 0;

  for(j1 = 0; j1 <= twojmax; j1++) {
    if(diagonalstyle == 0) {
      for(j2 = 0; j2 <= j1; j2++)
        for(j = abs(j1 - j2);
            j <= MIN(twojmax, j1 + j2); j += 2) {
          dbvec(ncount,0) = dbarray(j1,j2,j,0);
          dbvec(ncount,1) = dbarray(j1,j2,j,1);
          dbvec(ncount,2) = dbarray(j1,j2,j,2);
          ncount++;
        }
    } else if(diagonalstyle == 1) {
      j2 = j1;
      for(j = abs(j1 - j2);
          j <= MIN(twojmax, j1 + j2); j += 2) {
        dbvec(ncount,0) = dbarray(j1,j2,j,0);
        dbvec(ncount,1) = dbarray(j1,j2,j,1);
        dbvec(ncount,2) = dbarray(j1,j2,j,2);
        ncount++;
      }
    } else if(diagonalstyle == 2) {
      j = j2 = j1;
      dbvec(ncount,0) = dbarray(j1,j2,j,0);
      dbvec(ncount,1) = dbarray(j1,j2,j,1);
      dbvec(ncount,2) = dbarray(j1,j2,j,2);
      ncount++;
    } else if(diagonalstyle == 3) {
      for(j2 = 0; j2 <= j1; j2++)
        for(j = abs(j1 - j2);
            j <= MIN(twojmax, j1 + j2); j += 2)
	  if (j >= j1) {
	    dbvec(ncount,0) = dbarray(j1,j2,j,0);
	    dbvec(ncount,1) = dbarray(j1,j2,j,1);
	    dbvec(ncount,2) = dbarray(j1,j2,j,2);
	    ncount++;
	  }
    }
  }
}

/* ---------------------------------------------------------------------- */

void SNA::zero_uarraytot()
{
  for (int j = 0; j <= twojmax; j++)
    for (int ma = 0; ma <= j; ma++)
      for (int mb = 0; mb <= j; mb++) {
        uarraytot_r(j,ma,mb) = 0.0;
        uarraytot_i(j,ma,mb) = 0.0;
      }
}

/* ---------------------------------------------------------------------- */

void SNA::addself_uarraytot(double wself_in)
{
  for (int j = 0; j <= twojmax; j++)
    for (int ma = 0; ma <= j; ma++) {
      uarraytot_r(j,ma,ma) = wself_in;
      uarraytot_i(j,ma,ma) = 0.0;
    }
}

/* ----------------------------------------------------------------------
   add Wigner U-functions for one neighbor to the total
------------------------------------------------------------------------- */

void SNA::add_uarraytot(double r, double wj, double rcut)
{
  double sfac;

  sfac = compute_sfac(r, rcut);

  sfac *= wj;

  for (int j = 0; j <= twojmax; j++)
    for (int ma = 0; ma <= j; ma++)
      for (int mb = 0; mb <= j; mb++) {
        uarraytot_r(j,ma,mb) +=
          sfac * uarray_r(j,ma,mb);
        uarraytot_i(j,ma,mb) +=
          sfac * uarray_i(j,ma,mb);
      }
}

/* ----------------------------------------------------------------------
   compute Wigner U-functions for one neighbor
------------------------------------------------------------------------- */

void SNA::compute_uarray(double x, double y, double z,
                         double z0, double r)
{
  double r0inv;
  double a_r, b_r, a_i, b_i;
  double rootpq;

  // compute Cayley-Klein parameters for unit quaternion

  r0inv = 1.0 / sqrt(r * r + z0 * z0);
  a_r = r0inv * z0;
  a_i = -r0inv * z;
  b_r = r0inv * y;
  b_i = -r0inv * x;

  // VMK Section 4.8.2

  uarray_r(0,0,0) = 1.0;
  uarray_i(0,0,0) = 0.0;

  for (int j = 1; j <= twojmax; j++) {

    // fill in left side of matrix layer from previous layer

    for (int mb = 0; 2*mb <= j; mb++) {
      uarray_r(j,0,mb) = 0.0;
      uarray_i(j,0,mb) = 0.0;

      for (int ma = 0; ma < j; ma++) {
	rootpq = rootpqarray(j - ma,j - mb);
        uarray_r(j,ma,mb) +=
          rootpq *
          (a_r * uarray_r(j - 1,ma,mb) +
	   a_i * uarray_i(j - 1,ma,mb));
        uarray_i(j,ma,mb) +=
          rootpq *
          (a_r * uarray_i(j - 1,ma,mb) -
	   a_i * uarray_r(j - 1,ma,mb));

	rootpq = rootpqarray(ma + 1,j - mb);
        uarray_r(j,ma + 1,mb) =
          -rootpq *
          (b_r * uarray_r(j - 1,ma,mb) +
	   b_i * uarray_i(j - 1,ma,mb));
        uarray_i(j,ma + 1,mb) =
          -rootpq *
          (b_r * uarray_i(j - 1,ma,mb) -
	   b_i * uarray_r(j - 1,ma,mb));
      }
    }

    // copy left side to right side with inversion symmetry VMK 4.4(2)
    // u[ma-j,mb-j] = (-1)^(ma-mb)*Conj([u[ma,mb))

    int mbpar = -1;
    for (int mb = 0; 2*mb <= j; mb++) {
      mbpar = -mbpar;
      int mapar = -mbpar;
      for (int ma = 0; ma <= j; ma++) {
    	mapar = -mapar;
    	if (mapar == 1) {
    	  uarray_r(j,j-ma,j-mb) = uarray_r(j,ma,mb);
    	  uarray_i(j,j-ma,j-mb) = -uarray_i(j,ma,mb);
    	} else {
    	  uarray_r(j,j-ma,j-mb) = -uarray_r(j,ma,mb);
    	  uarray_i(j,j-ma,j-mb) = uarray_i(j,ma,mb);
    	}
    	//OK
    	//printf("%lf %lf %lf %lf %lf %lf %lf SNAP-COMPARE: UARRAY\n",x,y,z,z0,r,uarray_r(j,ma,mb),uarray_i(j,ma,mb));
      }
    }
  }
}


/* ----------------------------------------------------------------------
   compute derivatives of Wigner U-functions for one neighbor
   see comments in compute_uarray()
------------------------------------------------------------------------- */

void SNA::compute_duarray(double x, double y, double z,
                          double z0, double r, double dz0dr,
			  double wj, double rcut)
{
  double r0inv;
  double a_r, a_i, b_r, b_i;
  double da_r[3], da_i[3], db_r[3], db_i[3];
  double dz0[3], dr0inv[3], dr0invdr;
  double rootpq;

  double rinv = 1.0 / r;
  double ux = x * rinv;
  double uy = y * rinv;
  double uz = z * rinv;

  r0inv = 1.0 / sqrt(r * r + z0 * z0);
  a_r = z0 * r0inv;
  a_i = -z * r0inv;
  b_r = y * r0inv;
  b_i = -x * r0inv;

  dr0invdr = -pow(r0inv, 3.0) * (r + z0 * dz0dr);

  dr0inv[0] = dr0invdr * ux;
  dr0inv[1] = dr0invdr * uy;
  dr0inv[2] = dr0invdr * uz;

  dz0[0] = dz0dr * ux;
  dz0[1] = dz0dr * uy;
  dz0[2] = dz0dr * uz;

  for (int k = 0; k < 3; k++) {
    da_r[k] = dz0[k] * r0inv + z0 * dr0inv[k];
    da_i[k] = -z * dr0inv[k];
  }

  da_i[2] += -r0inv;

  for (int k = 0; k < 3; k++) {
    db_r[k] = y * dr0inv[k];
    db_i[k] = -x * dr0inv[k];
  }

  db_i[0] += -r0inv;
  db_r[1] += r0inv;

  uarray_r(0,0,0) = 1.0;
  duarray_r(0,0,0,0) = 0.0;
  duarray_r(0,0,0,1) = 0.0;
  duarray_r(0,0,0,2) = 0.0;
  uarray_i(0,0,0) = 0.0;
  duarray_i(0,0,0,0) = 0.0;
  duarray_i(0,0,0,1) = 0.0;
  duarray_i(0,0,0,2) = 0.0;

  for (int j = 1; j <= twojmax; j++) {
    for (int mb = 0; 2*mb <= j; mb++) {
      uarray_r(j,0,mb) = 0.0;
      duarray_r(j,0,mb,0) = 0.0;
      duarray_r(j,0,mb,1) = 0.0;
      duarray_r(j,0,mb,2) = 0.0;
      uarray_i(j,0,mb) = 0.0;
      duarray_i(j,0,mb,0) = 0.0;
      duarray_i(j,0,mb,1) = 0.0;
      duarray_i(j,0,mb,2) = 0.0;

      for (int ma = 0; ma < j; ma++) {
        rootpq = rootpqarray(j - ma,j - mb);
        uarray_r(j,ma,mb) += rootpq *
                               (a_r *  uarray_r(j - 1,ma,mb) +
                                a_i *  uarray_i(j - 1,ma,mb));
        uarray_i(j,ma,mb) += rootpq *
                               (a_r *  uarray_i(j - 1,ma,mb) -
                                a_i *  uarray_r(j - 1,ma,mb));

        for (int k = 0; k < 3; k++) {
          duarray_r(j,ma,mb,k) +=
            rootpq * (da_r[k] * uarray_r(j - 1,ma,mb) +
                      da_i[k] * uarray_i(j - 1,ma,mb) +
                      a_r * duarray_r(j - 1,ma,mb,k) +
                      a_i * duarray_i(j - 1,ma,mb,k));
          duarray_i(j,ma,mb,k) +=
            rootpq * (da_r[k] * uarray_i(j - 1,ma,mb) -
                      da_i[k] * uarray_r(j - 1,ma,mb) +
                      a_r * duarray_i(j - 1,ma,mb,k) -
                      a_i * duarray_r(j - 1,ma,mb,k));
        }

	rootpq = rootpqarray(ma + 1,j - mb);
        uarray_r(j,ma + 1,mb) =
          -rootpq * (b_r *  uarray_r(j - 1,ma,mb) +
                     b_i *  uarray_i(j - 1,ma,mb));
        uarray_i(j,ma + 1,mb) =
          -rootpq * (b_r *  uarray_i(j - 1,ma,mb) -
                     b_i *  uarray_r(j - 1,ma,mb));

        for (int k = 0; k < 3; k++) {
          duarray_r(j,ma + 1,mb,k) =
            -rootpq * (db_r[k] * uarray_r(j - 1,ma,mb) +
                       db_i[k] * uarray_i(j - 1,ma,mb) +
                       b_r * duarray_r(j - 1,ma,mb,k) +
                       b_i * duarray_i(j - 1,ma,mb,k));
          duarray_i(j,ma + 1,mb,k) =
            -rootpq * (db_r[k] * uarray_i(j - 1,ma,mb) -
                       db_i[k] * uarray_r(j - 1,ma,mb) +
                       b_r * duarray_i(j - 1,ma,mb,k) -
                       b_i * duarray_r(j - 1,ma,mb,k));
        }
      }
    }

    int mbpar = -1;
    for (int mb = 0; 2*mb <= j; mb++) {
      mbpar = -mbpar;
      int mapar = -mbpar;
      for (int ma = 0; ma <= j; ma++) {
    	mapar = -mapar;
    	if (mapar == 1) {
    	  uarray_r(j,j-ma,j-mb) = uarray_r(j,ma,mb);
    	  uarray_i(j,j-ma,j-mb) = -uarray_i(j,ma,mb);
    	  for (int k = 0; k < 3; k++) {
    	    duarray_r(j,j-ma,j-mb,k) = duarray_r(j,ma,mb,k);
    	    duarray_i(j,j-ma,j-mb,k) = -duarray_i(j,ma,mb,k);
    	  }
    	} else {
    	  uarray_r(j,j-ma,j-mb) = -uarray_r(j,ma,mb);
    	  uarray_i(j,j-ma,j-mb) = uarray_i(j,ma,mb);
    	  for (int k = 0; k < 3; k++) {
    	    duarray_r(j,j-ma,j-mb,k) = -duarray_r(j,ma,mb,k);
    	    duarray_i(j,j-ma,j-mb,k) = duarray_i(j,ma,mb,k);
    	  }
    	}
      }
    }
  }

  double sfac = compute_sfac(r, rcut);
  double dsfac = compute_dsfac(r, rcut);

  sfac *= wj;
  dsfac *= wj;

  for (int j = 0; j <= twojmax; j++)
    for (int ma = 0; ma <= j; ma++)
      for (int mb = 0; mb <= j; mb++) {
        duarray_r(j,ma,mb,0) = dsfac * uarray_r(j,ma,mb) * ux +
                                  sfac * duarray_r(j,ma,mb,0);
        duarray_i(j,ma,mb,0) = dsfac * uarray_i(j,ma,mb) * ux +
                                  sfac * duarray_i(j,ma,mb,0);
        duarray_r(j,ma,mb,1) = dsfac * uarray_r(j,ma,mb) * uy +
                                  sfac * duarray_r(j,ma,mb,1);
        duarray_i(j,ma,mb,1) = dsfac * uarray_i(j,ma,mb) * uy +
                                  sfac * duarray_i(j,ma,mb,1);
        duarray_r(j,ma,mb,2) = dsfac * uarray_r(j,ma,mb) * uz +
                                  sfac * duarray_r(j,ma,mb,2);
        duarray_i(j,ma,mb,2) = dsfac * uarray_i(j,ma,mb) * uz +
                                  sfac * duarray_i(j,ma,mb,2);
      }
}

/* ---------------------------------------------------------------------- */

void SNA::create_twojmax_arrays()
{
  int jdim = twojmax + 1;
  cgarray = t_sna_5d("sna:cgarray",jdim,jdim,jdim,jdim,jdim);
  rootpqarray = t_sna_2d("sna:barray",jdim+1,jdim+1);
  barray = t_sna_3d("sna:barray",jdim,jdim,jdim);
  dbarray = t_sna_4d("sna:dbarray",jdim,jdim,jdim);

  uarray_r = t_sna_3d("sna:uarray_r",jdim,jdim,jdim);
  uarray_i = t_sna_3d("sna:uarray_i",jdim,jdim,jdim);
  duarray_r = t_sna_4d("sna:duarray_r",jdim,jdim,jdim);
  duarray_i = t_sna_4d("sna:duarray_i",jdim,jdim,jdim);

  uarraytot_r = t_sna_3d("sna:uarraytot_r",jdim,jdim,jdim);
  uarraytot_i = t_sna_3d("sna:uarraytot_i",jdim,jdim,jdim);
  zarray_r = t_sna_5d("sna:zarray_r",jdim,jdim,jdim,jdim,jdim);
  zarray_i = t_sna_5d("sna:zarray_i",jdim,jdim,jdim,jdim,jdim);

}

/* ---------------------------------------------------------------------- */

void SNA::destroy_twojmax_arrays()
{
}

/* ----------------------------------------------------------------------
   factorial n
------------------------------------------------------------------------- */

double SNA::factorial(int n)
{
  double result = 1.0;
  for(int i=1; i<=n; i++)
    result *= 1.0*i;
  return result;
}

/* ----------------------------------------------------------------------
   the function delta given by VMK Eq. 8.2(1)
------------------------------------------------------------------------- */

double SNA::deltacg(int j1, int j2, int j)
{
  double sfaccg = factorial((j1 + j2 + j) / 2 + 1);
  return sqrt(factorial((j1 + j2 - j) / 2) *
              factorial((j1 - j2 + j) / 2) *
              factorial((-j1 + j2 + j) / 2) / sfaccg);
}

/* ----------------------------------------------------------------------
   assign Clebsch-Gordan coefficients using
   the quasi-binomial formula VMK 8.2.1(3)
------------------------------------------------------------------------- */

void SNA::init_clebsch_gordan()
{
  double sum,dcg,sfaccg;
  int m, aa2, bb2, cc2;
  int ifac;

  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= twojmax; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        for (int m1 = 0; m1 <= j1; m1 += 1) {
          aa2 = 2 * m1 - j1;

          for (int m2 = 0; m2 <= j2; m2 += 1) {

            // -c <= cc <= c

            bb2 = 2 * m2 - j2;
            m = (aa2 + bb2 + j) / 2;

            if(m < 0 || m > j) continue;

	    sum = 0.0;

	    for (int z = MAX(0, MAX(-(j - j2 + aa2)
				   / 2, -(j - j1 - bb2) / 2));
		z <= MIN((j1 + j2 - j) / 2,
			 MIN((j1 - aa2) / 2, (j2 + bb2) / 2));
		z++) {
	      ifac = z % 2 ? -1 : 1;
	      sum += ifac /
		(factorial(z) *
		 factorial((j1 + j2 - j) / 2 - z) *
		 factorial((j1 - aa2) / 2 - z) *
		 factorial((j2 + bb2) / 2 - z) *
		 factorial((j - j2 + aa2) / 2 + z) *
		 factorial((j - j1 - bb2) / 2 + z));
	    }

	    cc2 = 2 * m - j;
	    dcg = deltacg(j1, j2, j);
	    sfaccg = sqrt(factorial((j1 + aa2) / 2) *
			factorial((j1 - aa2) / 2) *
			factorial((j2 + bb2) / 2) *
			factorial((j2 - bb2) / 2) *
			factorial((j  + cc2) / 2) *
			factorial((j  - cc2) / 2) *
			(j + 1));

	    cgarray(j1,j2,j,m1,m2) = sum * dcg * sfaccg;
	    //printf("SNAP-COMPARE: CG: %i %i %i %i %i %e\n",j1,j2,j,m1,m2,cgarray(j1,j2,j,m1,m2));
	  }
	}
}

/* ----------------------------------------------------------------------
   pre-compute table of sqrt[p/m2], p, q = 1,twojmax
   the p = 0, q = 0 entries are allocated and skipped for convenience.
------------------------------------------------------------------------- */

void SNA::init_rootpqarray()
{
  for (int p = 1; p <= twojmax; p++)
    for (int q = 1; q <= twojmax; q++)
      rootpqarray(p,q) = sqrt(static_cast<double>(p)/q);
}


/* ---------------------------------------------------------------------- */

int SNA::compute_ncoeff()
{
  int ncount;

  ncount = 0;

  for (int j1 = 0; j1 <= twojmax; j1++)
    if(diagonalstyle == 0) {
      for (int j2 = 0; j2 <= j1; j2++)
        for (int j = abs(j1 - j2);
            j <= MIN(twojmax, j1 + j2); j += 2)
          ncount++;
    } else if(diagonalstyle == 1) {
      int j2 = j1;

      for (int j = abs(j1 - j2);
          j <= MIN(twojmax, j1 + j2); j += 2)
        ncount++;
    } else if(diagonalstyle == 2) {
      ncount++;
    } else if(diagonalstyle == 3) {
      for (int j2 = 0; j2 <= j1; j2++)
        for (int j = abs(j1 - j2);
            j <= MIN(twojmax, j1 + j2); j += 2)
          if (j >= j1) ncount++;
    }

  return ncount;
}

/* ---------------------------------------------------------------------- */

double SNA::compute_sfac(double r, double rcut)
{
  if (switch_flag == 0) return 1.0;
  if (switch_flag == 1) {
    if(r <= rmin0) return 1.0;
    else if(r > rcut) return 0.0;
    else {
      double rcutfac = MY_PI / (rcut - rmin0);
      return 0.5 * (cos((r - rmin0) * rcutfac) + 1.0);
    }
  }
  return 0.0;
}

/* ---------------------------------------------------------------------- */

double SNA::compute_dsfac(double r, double rcut)
{
  if (switch_flag == 0) return 0.0;
  if (switch_flag == 1) {
    if(r <= rmin0) return 0.0;
    else if(r > rcut) return 0.0;
    else {
      double rcutfac = MY_PI / (rcut - rmin0);
      return -0.5 * sin((r - rmin0) * rcutfac) * rcutfac;
    }
  }
  return 0.0;
}

