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

#ifdef MODULES_OPTION_CHECK
      if( (strcmp(argv[i+1], "CELL_FULL") == 0) )
        force_iteration_type = FORCE_ITER_CELL_FULL;
#endif
#ifdef MODULES_INSTANTIATION
    else if ((input->force_type == FORCE_LJ) && (input->force_iteration_type == FORCE_ITER_CELL_FULL)){
      force = new ForceLJCell(input->input_data.words[input->force_line],system,false);
    }
#endif


#if !defined(MODULES_OPTION_CHECK) && !defined(MODULES_INSTANTIATION)

#ifndef FORCE_LJ_CELL_H
#define FORCE_LJ_CELL_H
#include<force.h>

class ForceLJCell: public Force {
private:
  t_x x;
  t_f f;
  t_id id;
  t_type type;
  Binning::t_bincount bin_count;
  Binning::t_binoffsets bin_offsets;
  Binning::t_permute_vector permute_vector;
  T_INT nbinx,nbiny,nbinz,nhalo;
  int N_local;
  int step;

  typedef Kokkos::View<T_F_FLOAT**> t_fparams;
  t_fparams lj1,lj2,cutsq;

public:
  struct TagCell {};
  struct TagCellPE {};

  typedef Kokkos::TeamPolicy<TagCell, Kokkos::IndexType<T_INT> > t_policy;
  typedef Kokkos::TeamPolicy<TagCellPE, Kokkos::IndexType<T_INT> > t_policy_pe;

  ForceLJCell (char** args, System* system, bool half_neigh);

  void init_coeff(int nargs, char** args);

  void compute(System* system, Binning* binning, Neighbor* );
  T_F_FLOAT compute_energy(System* system, Binning* binning, Neighbor* );

  KOKKOS_INLINE_FUNCTION
    void operator() (TagCell, const typename t_policy::member_type& team) const {
    const T_INT bx = team.league_rank()/(nbiny*nbinz);
    const T_INT by = (team.league_rank()/(nbinz)) % nbiny;
    const T_INT bz = team.league_rank() % nbinz;

    const T_INT i_offset = bin_offsets(bx,by,bz);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,bin_count(bx,by,bz)), [&] (const int bi) {
      const T_INT i = permute_vector(i_offset + bi);
      if(i>=N_local) return;
      const T_F_FLOAT x_i = x(i,0);
      const T_F_FLOAT y_i = x(i,1);
      const T_F_FLOAT z_i = x(i,2);
      const int type_i = type(i);

      t_scalar3<T_F_FLOAT> f_i;
      for(int bx_j = bx>0?bx-1:bx; bx_j < (bx+1<nbinx?bx+2:bx+1); bx_j++)
      for(int by_j = by>0?by-1:by; by_j < (by+1<nbiny?by+2:by+1); by_j++)
      for(int bz_j = bz>0?bz-1:bz; bz_j < (bz+1<nbinz?bz+2:bz+1); bz_j++) {

        const T_INT j_offset = bin_offsets(bx_j,by_j,bz_j);

        t_scalar3<T_F_FLOAT> f_i_tmp;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, bin_count(bx_j,by_j,bz_j)), [&]
          (const T_INT bj, t_scalar3<T_F_FLOAT>& lf_i) {
          T_INT j = permute_vector(j_offset + bj);
          const T_F_FLOAT dx = x_i - x(j,0);
          const T_F_FLOAT dy = y_i - x(j,1);
          const T_F_FLOAT dz = z_i - x(j,2);
          
          const int type_j = type(j);
          const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;
          
          if((rsq < cutsq(type_i,type_j)) && (i!=j)) {
            T_F_FLOAT r2inv = 1.0/rsq;
            T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
            T_F_FLOAT fpair = (r6inv * (lj1(type_i,type_j)*r6inv - lj2(type_i,type_j))) * r2inv;
            lf_i.x += dx*fpair;
            lf_i.y += dy*fpair;
            lf_i.z += dz*fpair;
          }
        },f_i_tmp);
        f_i += f_i_tmp;
      }
      f(i,0) += f_i.x;
      f(i,1) += f_i.y;
      f(i,2) += f_i.z;
    });
  } 

  KOKKOS_INLINE_FUNCTION
  void operator() (TagCellPE, const typename t_policy_pe::member_type& team, T_F_FLOAT& PE_bi) const {
    const T_INT bx = team.league_rank()/(nbiny*nbinz);
    const T_INT by = (team.league_rank()/(nbinz)) % nbiny;
    const T_INT bz = team.league_rank() % nbinz;
    const bool shift_flag = true;

    T_F_FLOAT PE_i;
    const T_INT i_offset = bin_offsets(bx,by,bz);
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,0,bin_count(bx,by,bz)), [&]
      (const int bi, T_F_FLOAT& PE_i) {
      const T_INT i = permute_vector(i_offset + bi);
      if(i>=N_local) return;
      const T_F_FLOAT x_i = x(i,0);
      const T_F_FLOAT y_i = x(i,1);
      const T_F_FLOAT z_i = x(i,2);
      const int type_i = type(i);

      for(int bx_j = bx>0?bx-1:bx; bx_j < (bx+1<nbinx?bx+2:bx+1); bx_j++)
      for(int by_j = by>0?by-1:by; by_j < (by+1<nbiny?by+2:by+1); by_j++)
      for(int bz_j = bz>0?bz-1:bz; bz_j < (bz+1<nbinz?bz+2:bz+1); bz_j++) {

        const T_INT j_offset = bin_offsets(bx_j,by_j,bz_j);

        T_F_FLOAT PE_ibj;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, bin_count(bx_j,by_j,bz_j)), [&]
          (const T_INT bj, T_F_FLOAT& PE_ibj) {
          T_INT j = permute_vector(j_offset + bj);
          const T_F_FLOAT dx = x_i - x(j,0);
          const T_F_FLOAT dy = y_i - x(j,1);
          const T_F_FLOAT dz = z_i - x(j,2);
          
          const int type_j = type(j);
          const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;
          
          if((rsq < cutsq(type_i,type_j)) && (i!=j)) {
            T_F_FLOAT r2inv = 1.0/rsq;
            T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
            PE_ibj += 0.5*r6inv * (0.5*lj1(type_i,type_j)*r6inv - lj2(type_i,type_j)) / 6.0; // optimize later
            if (shift_flag) {
              T_F_FLOAT r2invc = 1.0/cutsq(type_i,type_j);
              T_F_FLOAT r6invc = r2invc*r2invc*r2invc;
              PE_ibj -= 0.5*r6invc * (0.5*lj1(type_i,type_j)*r6invc - lj2(type_i,type_j)) / 6.0; // optimize later
            }
          }
        },PE_ibj);
        PE_i += PE_ibj;
      }
      },PE_i);
    PE_bi += PE_i;
  }

  const char* name();
};
#endif
#endif
