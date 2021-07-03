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

#include<force_lj_neigh.h>

template<class NeighborClass>
ForceLJNeigh<NeighborClass>::ForceLJNeigh(char** args, System* system, bool half_neigh_):Force(args,system,half_neigh_) {
  ntypes = system->ntypes;
  use_stackparams = (ntypes <= MAX_TYPES_STACKPARAMS);
  if (!use_stackparams) {
    lj1 = t_fparams("ForceLJNeigh::lj1",ntypes,ntypes);
    lj2 = t_fparams("ForceLJNeigh::lj2",ntypes,ntypes);
    cutsq = t_fparams("ForceLJNeigh::cutsq",ntypes,ntypes);
  }
  nbinx = nbiny = nbinz = 0;
  N_local = 0;
  nhalo = 0;
  step = 0;
}

template<class NeighborClass>
void ForceLJNeigh<NeighborClass>::init_coeff(int nargs, char** args) {
  step = 0;

  int one_based_type = 1;
  int t1 = atoi(args[1])-one_based_type;
  int t2 = atoi(args[2])-one_based_type;
  double eps = atof(args[3]);
  double sigma = atof(args[4]);
  double cut = atof(args[5]);

  if (use_stackparams) {
    for (int i = 0; i < ntypes; i++) {
      for (int j = 0; j < ntypes; j++) {
        stack_lj1[i][j] = 48.0 * eps * pow(sigma,12.0);
        stack_lj2[i][j] = 24.0 * eps * pow(sigma,6.0);
        stack_cutsq[i][j] = cut*cut;
      }
    }
  } else {
    t_fparams::HostMirror h_lj1 = Kokkos::create_mirror_view(lj1);
    t_fparams::HostMirror h_lj2 = Kokkos::create_mirror_view(lj2);
    t_fparams::HostMirror h_cutsq = Kokkos::create_mirror_view(cutsq);
    Kokkos::deep_copy(h_lj1,lj1);
    Kokkos::deep_copy(h_lj2,lj2);
    Kokkos::deep_copy(h_cutsq,cutsq);

    h_lj1(t1,t2) = 48.0 * eps * pow(sigma,12.0);
    h_lj2(t1,t2) = 24.0 * eps * pow(sigma,6.0);
    h_lj1(t2,t1) = h_lj1(t1,t2);
    h_lj2(t2,t1) = h_lj2(t1,t2);
    h_cutsq(t1,t2) = cut*cut;
    h_cutsq(t2,t1) = cut*cut;

    Kokkos::deep_copy(lj1,h_lj1);
    Kokkos::deep_copy(lj2,h_lj2);
    Kokkos::deep_copy(cutsq,h_cutsq);

    rnd_lj1 = lj1;
    rnd_lj2 = lj2;
    rnd_cutsq = cutsq;
  }
};

template<class NeighborClass>
void ForceLJNeigh<NeighborClass>::compute(System* system, Binning* binning, Neighbor* neighbor_ ) {
  // Set internal data handles
  NeighborClass* neighbor = (NeighborClass*) neighbor_;
  neigh_list = neighbor->get_neigh_list();

  N_local = system->N_local;
  x = system->x;
  f = system->f;
  //f_a = system->f;
  f_r = Kokkos::Experimental::create_scatter_view<>(system->f);
  type = system->type;
  id = system->id;
  if (use_stackparams) {
    if(half_neigh)
      Kokkos::parallel_for("ForceLJNeigh::compute", t_policy_half_neigh_stackparams(0, system->N_local), *this);
    else
      Kokkos::parallel_for("ForceLJNeigh::compute", t_policy_full_neigh_stackparams(0, system->N_local), *this);
  } else {
    if(half_neigh)
      Kokkos::parallel_for("ForceLJNeigh::compute", t_policy_half_neigh(0, system->N_local), *this);
    else
      Kokkos::parallel_for("ForceLJNeigh::compute", t_policy_full_neigh(0, system->N_local), *this);
  }
  Kokkos::fence();

  if(half_neigh) {
    Kokkos::Experimental::contribute(system->f, f_r);
    f_r = decltype(f_r)();
  }

  step++;
}

template<class NeighborClass>
T_V_FLOAT ForceLJNeigh<NeighborClass>::compute_energy(System* system, Binning* binning, Neighbor* neighbor_ ) {
  // Set internal data handles
  NeighborClass* neighbor = (NeighborClass*) neighbor_;
  neigh_list = neighbor->get_neigh_list();

  N_local = system->N_local;
  x = system->x;
  type = system->type;
  id = system->id;
  T_V_FLOAT energy;
  if (use_stackparams) {
    if(half_neigh)
      Kokkos::parallel_reduce("ForceLJNeigh::compute_energy", t_policy_half_neigh_pe_stackparams(0, system->N_local), *this, energy);
    else
      Kokkos::parallel_reduce("ForceLJNeigh::compute_energy", t_policy_full_neigh_pe_stackparams(0, system->N_local), *this, energy);
  } else {
    if(half_neigh)
      Kokkos::parallel_reduce("ForceLJNeigh::compute_energy", t_policy_half_neigh_pe(0, system->N_local), *this, energy);
    else
      Kokkos::parallel_reduce("ForceLJNeigh::compute_energy", t_policy_full_neigh_pe(0, system->N_local), *this, energy);
  }
  Kokkos::fence();

  step++;
  return energy;
}

template<class NeighborClass>
const char* ForceLJNeigh<NeighborClass>::name() { return half_neigh?"ForceLJNeighHalf":"ForceLJNeighFull"; }

template<class NeighborClass>
template<bool STACKPARAMS>
KOKKOS_INLINE_FUNCTION
void ForceLJNeigh<NeighborClass>::operator() (TagFullNeigh<STACKPARAMS>, const T_INT& i) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);

  typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);

  const int num_neighs = neighs_i.get_num_neighs();

  T_F_FLOAT fxi = 0.0;
  T_F_FLOAT fyi = 0.0;
  T_F_FLOAT fzi = 0.0;

  for(int jj = 0; jj < num_neighs; jj++) {
    T_INT j = neighs_i(jj);
    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    const T_F_FLOAT cutsq_ij = STACKPARAMS?stack_cutsq[type_i][type_j]:rnd_cutsq(type_i,type_j);

    if( rsq < cutsq_ij ) {
      const T_F_FLOAT lj1_ij = STACKPARAMS?stack_lj1[type_i][type_j]:rnd_lj1(type_i,type_j);
      const T_F_FLOAT lj2_ij = STACKPARAMS?stack_lj2[type_i][type_j]:rnd_lj2(type_i,type_j);

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      T_F_FLOAT fpair = (r6inv * (lj1_ij*r6inv - lj2_ij)) * r2inv;
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
template<bool STACKPARAMS>
KOKKOS_INLINE_FUNCTION
void ForceLJNeigh<NeighborClass>::operator() (TagHalfNeigh<STACKPARAMS>, const T_INT& i) const {
  auto f_a = f_r.access();

  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);

  typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);

  const int num_neighs = neighs_i.get_num_neighs();

  T_F_FLOAT fxi = 0.0;
  T_F_FLOAT fyi = 0.0;
  T_F_FLOAT fzi = 0.0;
  for(int jj = 0; jj < num_neighs; jj++) {
    T_INT j = neighs_i(jj);
    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    const T_F_FLOAT cutsq_ij = STACKPARAMS?stack_cutsq[type_i][type_j]:rnd_cutsq(type_i,type_j);

    if( rsq < cutsq_ij ) {
      const T_F_FLOAT lj1_ij = STACKPARAMS?stack_lj1[type_i][type_j]:rnd_lj1(type_i,type_j);
      const T_F_FLOAT lj2_ij = STACKPARAMS?stack_lj2[type_i][type_j]:rnd_lj2(type_i,type_j);

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      T_F_FLOAT fpair = (r6inv * (lj1_ij*r6inv - lj2_ij)) * r2inv;
      fxi += dx*fpair;
      fyi += dy*fpair;
      fzi += dz*fpair;
      f_a(j,0) -= dx*fpair;
      f_a(j,1) -= dy*fpair;
      f_a(j,2) -= dz*fpair;
    }
  }
  f_a(i,0) += fxi;
  f_a(i,1) += fyi;
  f_a(i,2) += fzi;

}

template<class NeighborClass>
template<bool STACKPARAMS>
KOKKOS_INLINE_FUNCTION
void ForceLJNeigh<NeighborClass>::operator() (TagFullNeighPE<STACKPARAMS>, const T_INT& i, T_V_FLOAT& PE) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);
  const bool shift_flag = true;

  typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);

  const int num_neighs = neighs_i.get_num_neighs();

  for(int jj = 0; jj < num_neighs; jj++) {
    T_INT j = neighs_i(jj);
    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    const T_F_FLOAT cutsq_ij = STACKPARAMS?stack_cutsq[type_i][type_j]:rnd_cutsq(type_i,type_j);

    if( rsq < cutsq_ij ) {
      const T_F_FLOAT lj1_ij = STACKPARAMS?stack_lj1[type_i][type_j]:rnd_lj1(type_i,type_j);
      const T_F_FLOAT lj2_ij = STACKPARAMS?stack_lj2[type_i][type_j]:rnd_lj2(type_i,type_j);

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      PE += 0.5*r6inv * (0.5*lj1_ij*r6inv - lj2_ij) / 6.0; // optimize later

      if (shift_flag) {
        T_F_FLOAT r2invc = 1.0/cutsq_ij;
        T_F_FLOAT r6invc = r2invc*r2invc*r2invc;
        PE -= 0.5*r6invc * (0.5*lj1_ij*r6invc - lj2_ij) / 6.0; // optimize later
      }
    }
  }
}

template<class NeighborClass>
template<bool STACKPARAMS>
KOKKOS_INLINE_FUNCTION
void ForceLJNeigh<NeighborClass>::operator() (TagHalfNeighPE<STACKPARAMS>, const T_INT& i, T_V_FLOAT& PE) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);
  const bool shift_flag = true;

  typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);

  const int num_neighs = neighs_i.get_num_neighs();

  for(int jj = 0; jj < num_neighs; jj++) {
    T_INT j = neighs_i(jj);
    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    const T_F_FLOAT cutsq_ij = STACKPARAMS?stack_cutsq[type_i][type_j]:rnd_cutsq(type_i,type_j);

    if( rsq < cutsq_ij ) {
      const T_F_FLOAT lj1_ij = STACKPARAMS?stack_lj1[type_i][type_j]:rnd_lj1(type_i,type_j);
      const T_F_FLOAT lj2_ij = STACKPARAMS?stack_lj2[type_i][type_j]:rnd_lj2(type_i,type_j);

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      T_F_FLOAT fac;
      if(j<N_local) fac = 1.0;
      else fac = 0.5;

      PE += fac * r6inv * (0.5*lj1_ij*r6inv - lj2_ij) / 6.0;  // optimize later

      if (shift_flag) {
        T_F_FLOAT r2invc = 1.0/cutsq_ij;
        T_F_FLOAT r6invc = r2invc*r2invc*r2invc;
        PE -= fac * r6invc * (0.5*lj1_ij*r6invc - lj2_ij) / 6.0;  // optimize later
      }
    }
  }

}
