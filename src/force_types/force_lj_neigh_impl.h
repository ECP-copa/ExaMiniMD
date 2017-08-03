#include<force_lj_neigh.h>

template<class NeighborClass>
ForceLJNeigh<NeighborClass>::ForceLJNeigh(char** args, System* system, bool half_neigh_):Force(args,system,half_neigh_) {
  lj1 = t_fparams("ForceLJNeigh::lj1",system->ntypes,system->ntypes);
  lj2 = t_fparams("ForceLJNeigh::lj2",system->ntypes,system->ntypes);
  cutsq = t_fparams("ForceLJNeigh::cutsq",system->ntypes,system->ntypes);
  nbinx = nbiny = nbinz = 0;
  N_local = 0;
  nhalo = 0;
  step = 0;
}

template<class NeighborClass>
void ForceLJNeigh<NeighborClass>::init_coeff(int nargs, char** args) {
  int one_based_type = 1;
  int t1 = atoi(args[1])-one_based_type;
  int t2 = atoi(args[2])-one_based_type;
  double eps = atof(args[3]);
  double sigma = atof(args[4]);
  double cut = atof(args[5]);

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
  step = 0;
};

template<class NeighborClass>
void ForceLJNeigh<NeighborClass>::compute(System* system, Binning* binning, Neighbor* neighbor_ ) {
  // Set internal data handles
  NeighborClass* neighbor = (NeighborClass*) neighbor_;
  neigh_list = neighbor->get_neigh_list();

  N_local = system->N_local;
  x = system->x;
  f = system->f;
  f_a = system->f;
  type = system->type;
  id = system->id;
  if(half_neigh)
    Kokkos::parallel_for("ForceLJNeigh::computer", t_policy_half_neigh(0, system->N_local), *this);
  else
    Kokkos::parallel_for("ForceLJNeigh::computer", t_policy_full_neigh(0, system->N_local), *this);
  Kokkos::fence();

  // Reset internal data handles so we don't keep a reference count
  /*x = t_x();
  type = t_type();
  f = t_f();
  neigh_list = NeighborCSR<t_neigh_mem_space>::t_neigh_list();*/
  step++;
}

template<class NeighborClass>
const char* ForceLJNeigh<NeighborClass>::name() { return half_neigh?"ForceLJNeighHalf":"ForceLJNeighFull"; }

template<class NeighborClass>
KOKKOS_INLINE_FUNCTION
void ForceLJNeigh<NeighborClass>::operator() (TagFullNeigh, const T_INT& i) const {
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

    if( rsq < rnd_cutsq(type_i,type_j) ) {

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      T_F_FLOAT fpair = (r6inv * (rnd_lj1(type_i,type_j)*r6inv - rnd_lj2(type_i,type_j))) * r2inv;
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
void ForceLJNeigh<NeighborClass>::operator() (TagHalfNeigh, const T_INT& i) const {
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

    if( rsq < rnd_cutsq(type_i,type_j) ) {

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      T_F_FLOAT fpair = (r6inv * (rnd_lj1(type_i,type_j)*r6inv - rnd_lj2(type_i,type_j))) * r2inv;
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

