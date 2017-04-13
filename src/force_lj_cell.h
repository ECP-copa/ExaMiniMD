#ifdef MODULES_OPTION_CHECK
      if( (strcmp(argv[i+1], "CELL_FULL") == 0) )
        force_iteration_type = FORCE_ITER_CELL_FULL;
#endif
#ifdef MODULES_INSTANTIATION
    else if (input->force_type == FORCE_LJ_CELL) {
      force = new ForceLJCell(input->input_data.words[input->force_line],system);
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
  typedef Kokkos::TeamPolicy<Kokkos::IndexType<T_INT> > t_policy;

  ForceLJCell (char** args, System* system);

  void init_coeff(int nargs, char** args);

  void compute(System* system, Binning* binning, Neighbor* );

  KOKKOS_INLINE_FUNCTION
  void operator() (const typename t_policy::member_type& team) const {
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

      for(int bx_j = bx>0?bx-1:bx; bx_j < (bx+1<nbinx?bx+2:bx+1); bx_j++)
      for(int by_j = by>0?by-1:by; by_j < (by+1<nbiny?by+2:by+1); by_j++)
      for(int bz_j = bz>0?bz-1:bz; bz_j < (bz+1<nbinz?bz+2:bz+1); bz_j++) {

        const T_INT j_offset = bin_offsets(bx_j,by_j,bz_j);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, bin_count(bx_j,by_j,bz_j)), [&] (const T_INT bj) {
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
            f(i,0) += dx*fpair;
            f(i,1) += dy*fpair;
            f(i,2) += dz*fpair;
          }
        });
      }
    });
  } 
};
#endif
#endif
