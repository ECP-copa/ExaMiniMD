#include<force_lj_cell.h>

ForceLJCell::ForceLJCell(char** args, System* system, bool half_neigh_):Force(args,system,half_neigh) {
  lj1 = t_fparams("ForceLJCell::lj1",system->ntypes,system->ntypes);
  lj2 = t_fparams("ForceLJCell::lj2",system->ntypes,system->ntypes);
  cutsq = t_fparams("ForceLJCell::cutsq",system->ntypes,system->ntypes);
}

void ForceLJCell::init_coeff(int nargs, char** args) {
  int one_based_type = 1;
  int t1 = atoi(args[1])-one_based_type;
  int t2 = atoi(args[2])-one_based_type;
  double eps = atof(args[3]);
  double sigma = atof(args[4]);
  double cut = atof(args[5]);

  lj1(t1,t2) = 48.0 * eps * pow(sigma,12.0);
  lj2(t1,t2) = 24.0 * eps * pow(sigma,6.0);
  lj1(t2,t1) = lj1(t1,t2);
  lj2(t2,t1) = lj2(t1,t2);
  cutsq(t1,t2) = cut*cut;
  cutsq(t2,t1) = cut*cut;
};

void ForceLJCell::compute(System* system, Binning* binning, Neighbor*) {
  x = system->x;
  f = system->f;
  id = system->id;
  type = system->type;
  N_local = system->N_local;


  static int step_i = 0;
  step = step_i;
  bin_count = binning->bincount;
  bin_offsets = binning->binoffsets;
  permute_vector = binning->permute_vector;

  nhalo = binning->nhalo;
  nbinx = binning->nbinx;
  nbiny = binning->nbiny;
  nbinz = binning->nbinz;

  Kokkos::deep_copy(f,0.0);
  T_INT nbins = nbinx*nbiny*nbinz;

  Kokkos::parallel_for("ForceLJCell::computer", t_policy(nbins,1,8), *this);

  step_i++;
  x = t_x();
  type = t_type();
  f = t_f();

}
