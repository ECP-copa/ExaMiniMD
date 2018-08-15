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

#include<force_lj_idial_neigh.h>

template<class NeighborClass>
ForceLJIDialNeigh<NeighborClass>::ForceLJIDialNeigh(char** args, System* system, bool half_neigh_):Force(args,system,half_neigh_) {
  lj1 = t_fparams("ForceLJIDialNeigh::lj1",system->ntypes,system->ntypes);
  lj2 = t_fparams("ForceLJIDialNeigh::lj2",system->ntypes,system->ntypes);
  cutsq = t_fparams("ForceLJIDialNeigh::cutsq",system->ntypes,system->ntypes);
  intensity = t_fparams("ForceLJIDialNeigh::intensity",system->ntypes,system->ntypes);
}

template<class NeighborClass>
void ForceLJIDialNeigh<NeighborClass>::init_coeff(int nargs, char** args) {
  int one_based_type = 1;
  int t1 = atoi(args[1])-one_based_type;
  int t2 = atoi(args[2])-one_based_type;
  double eps = atof(args[3]);
  double sigma = atof(args[4]);
  double cut = atof(args[5]);
  int nrepeat = atoi(args[6]);

  t_fparams::HostMirror h_lj1 = Kokkos::create_mirror_view(lj1);
  t_fparams::HostMirror h_lj2 = Kokkos::create_mirror_view(lj2);
  t_fparams::HostMirror h_cutsq = Kokkos::create_mirror_view(cutsq);
  t_fparams::HostMirror h_intensity = Kokkos::create_mirror_view(intensity);
  Kokkos::deep_copy(h_lj1,lj1);
  Kokkos::deep_copy(h_lj2,lj2);
  Kokkos::deep_copy(h_cutsq,cutsq);
  Kokkos::deep_copy(h_intensity,intensity);

  h_lj1(t1,t2) = 48.0 * eps * pow(sigma,12.0);
  h_lj2(t1,t2) = 24.0 * eps * pow(sigma,6.0);
  h_lj1(t2,t1) = h_lj1(t1,t2);
  h_lj2(t2,t1) = h_lj2(t1,t2);
  h_cutsq(t1,t2) = cut*cut;
  h_cutsq(t2,t1) = cut*cut;
  h_intensity(t1,t2) = nrepeat;
  h_intensity(t2,t1) = nrepeat;

  Kokkos::deep_copy(lj1,h_lj1);
  Kokkos::deep_copy(lj2,h_lj2);
  Kokkos::deep_copy(cutsq,h_cutsq);
  Kokkos::deep_copy(intensity,h_intensity);

  rnd_lj1 = lj1;
  rnd_lj2 = lj2;
  rnd_cutsq = cutsq;
  rnd_intensity = intensity;
  step = 0;
};

template<class NeighborClass>
void ForceLJIDialNeigh<NeighborClass>::compute(System* system, Binning* binning, Neighbor* neighbor_ ) {
  // Set internal data handles
  NeighborClass* neighbor = (NeighborClass*) neighbor_;
  neigh_list = neighbor->get_neigh_list();

  N_local = system->N_local;
  x = system->x;
  f = system->f;
  f_a = system->f;
  type = system->type;
  if(half_neigh)
    Kokkos::parallel_for("ForceLJIDialNeigh::computer", t_policy_half_neigh(0, system->N_local), *this);
  else
    Kokkos::parallel_for("ForceLJIDialNeigh::computer", t_policy_full_neigh(0, system->N_local), *this);
  Kokkos::fence();

  // Reset internal data handles so we don't keep a reference count
  /*x = t_x();
  type = t_type();
  f = t_f();
  neigh_list = NeighborCSR<t_neigh_mem_space>::t_neigh_list();*/
  step++;
}

template<class NeighborClass>
const char* ForceLJIDialNeigh<NeighborClass>::name() { return half_neigh?"ForceLJIDialNeighHalf":"ForceLJIDialNeighFull"; }

template<class NeighborClass>
KOKKOS_INLINE_FUNCTION
void ForceLJIDialNeigh<NeighborClass>::operator() (TagFullNeigh, const T_INT& i) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);

  typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);

  const int num_neighs = neighs_i.get_num_neighs();

  T_F_FLOAT fxi = 0.0;
  T_F_FLOAT fyi = 0.0;
  T_F_FLOAT fzi = 0.0;

//    printf("NUMNEIGHS: %i %i\n",i,num_neighs);
  for(int jj = 0; jj < num_neighs; jj++) {
    T_INT j = neighs_i(jj);
    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    if( rsq < rnd_cutsq(type_i,type_j) ) {
      //-----------------
      // This 'for' loop increases the computational intensity of the LJ force evaluation.
      // intensity(type_i,type_j) acts as an intensity dial.
      // Could/should we implement this as a 'kokkos:parallel_for' loop?
      //-----------------
      T_F_FLOAT fpair = 0;
      for(int repeat = 0; repeat < intensity(type_i,type_j); repeat++) {
        T_F_FLOAT r2inv = 1.0/rsq;
        T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
        fpair += (r6inv * (rnd_lj1(type_i,type_j)*r6inv
                - rnd_lj2(type_i,type_j))) * r2inv/intensity(type_i,type_j);
      }
      fxi += dx*fpair;
      fyi += dy*fpair;
      fzi += dz*fpair;
    }
  }
  f(i,0) += fxi;
  f(i,1) += fyi;
  f(i,2) += fzi;

}

template<class NeighborClass>
KOKKOS_INLINE_FUNCTION
void ForceLJIDialNeigh<NeighborClass>::operator() (TagHalfNeigh, const T_INT& i) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);

  typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);

  const int num_neighs = neighs_i.get_num_neighs();

  T_F_FLOAT fxi = 0.0;
  T_F_FLOAT fyi = 0.0;
  T_F_FLOAT fzi = 0.0;
//printf("NUMNEIGHS: %i %i\n",i,num_neighs);
  for(int jj = 0; jj < num_neighs; jj++) {
    T_INT j = neighs_i(jj);
    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    if( rsq < rnd_cutsq(type_i,type_j) ) {
      //-----------------
      // This 'for' loop increases the computational intensity of the LJ force evaluation.
      // intensity(type_i,type_j) acts as an intensity dial.
      // Could/should we implement this as a 'kokkos:parallel_for' loop?
      //-----------------
      T_F_FLOAT fpair = 0;
      for(int repeat = 0; repeat < intensity(type_i,type_j); repeat++) {
        T_F_FLOAT r2inv = 1.0/rsq;
        T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
        fpair += (r6inv * (rnd_lj1(type_i,type_j)*r6inv
                  - rnd_lj2(type_i,type_j))) * r2inv/intensity(type_i,type_j);
      }
      fxi += dx*fpair;
      fyi += dy*fpair;
      fzi += dz*fpair;
      if(j<N_local) {
        f_a(j,0) -= dx*fpair;
        f_a(j,1) -= dy*fpair;
        f_a(j,2) -= dz*fpair;
      }
    }
  }
  f_a(i,0) += fxi;
  f_a(i,1) += fyi;
  f_a(i,2) += fzi;

}
