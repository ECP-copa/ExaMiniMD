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

T_F_FLOAT ForceLJCell::compute_energy(System* system, Binning* binning, Neighbor*) {
  x = system->x;
  id = system->id;
  type = system->type;
  N_local = system->N_local;

  bin_count = binning->bincount;
  bin_offsets = binning->binoffsets;
  permute_vector = binning->permute_vector;

  nhalo = binning->nhalo;
  nbinx = binning->nbinx;
  nbiny = binning->nbiny;
  nbinz = binning->nbinz;

  T_INT nbins = nbinx*nbiny*nbinz;
  T_F_FLOAT PE;
  Kokkos::parallel_reduce("ForceLJCell::compute_energy", t_policy_pe(nbins,1,8), *this, PE);

  x = t_x();
  id = t_id();
  type = t_type();

  return PE;
}

const char* ForceLJCell::name() { return "ForceLJCellFull"; }
