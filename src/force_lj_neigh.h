#ifdef MODULES_OPTION_CHECK
      if( (strcmp(argv[i+1], "NEIGH_FULL") == 0) )
        force_iteration_type = FORCE_ITER_NEIGH_FULL;
#endif
#ifdef MODULES_INSTANTIATION
    else if (input->force_type == FORCE_LJ_NEIGH) {
      force = new ForceLJNeigh(input->input_data.words[input->force_line],system);
    }
#endif


#if !defined(MODULES_OPTION_CHECK) && !defined(MODULES_INSTANTIATION)

#ifndef FORCE_LJ_NEIGH_H
#define FORCE_LJ_NEIGH_H
#include<force.h>

class ForceLJNeigh: public Force {
private:
  t_x x;
  t_f f;
  t_id id;
  t_type type;
  Binning::t_bincount bin_count;
  Binning::t_binoffsets bin_offsets;
  T_INT nbinx,nbiny,nbinz,nhalo;
  int step;

  typedef Kokkos::View<T_F_FLOAT**> t_fparams;
  t_fparams lj1,lj2,cutsq;
  typedef NeighListCSR<t_neigh_mem_space> t_neigh_list;

  t_neigh_list neigh_list;

public:
  typedef Kokkos::RangePolicy<Kokkos::IndexType<T_INT> > t_policy;

  ForceLJNeigh (char** args, System* system);

  void init_coeff(int nargs, char** args);

  void compute(System* system, Binning* binning, Neighbor* neighbor );

  /*
  KOKKOS_INLINE_FUNCTION
  void operator() (const typename t_policy::member_type& team) const {
    const T_INT bx = team.league_rank()/(nbiny*nbinz) + nhalo;
    const T_INT by = (team.league_rank()/(nbinz)) % nbiny + nhalo;
    const T_INT bz = team.league_rank() % nbinz + nhalo;

    const T_INT i_offset = bin_offsets(bx,by,bz);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,bin_count(bx,by,bz)), [&] (const int bi) {
      const T_INT i = i_offset + bi;
      const T_F_FLOAT x_i = x(i,0);
      const T_F_FLOAT y_i = x(i,1);
      const T_F_FLOAT z_i = x(i,2);
      const int type_i = type(i);

      typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);

      int count = 0;
      for(int bx_j = bx-1; bx_j<bx+2; bx_j++)
      for(int by_j = by-1; by_j<by+2; by_j++)
      for(int bz_j = bz-1; bz_j<bz+2; bz_j++) {

        const T_INT j_offset = bin_offsets(bx_j,by_j,bz_j);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, bin_count(bx_j,by_j,bz_j)), [&] (const T_INT bj) {
          T_INT j = j_offset + bj; 
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
          if(rsq < 9.0 )
            count++;
        });
      }
      printf("NumNeighs: %i %i %i\n",i,count,neighs_i.get_num_neighs());
    });
  } 
  */

  KOKKOS_INLINE_FUNCTION
  void operator() (const T_INT& i) const {
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
      //if(x_i*x_i<0.5 && y_i*y_i<0.5 && z_i*z_i<6.1 && z_i*z_i>2.0 )
      //  printf("DistanceForce %i: %lf %lf %lf %lf %lf\n",step,x(j,0),x(j,1),x(j,2),rsq,cutsq(type_i,type_j));

      if( rsq < cutsq(type_i,type_j) ) {

        T_F_FLOAT r2inv = 1.0/rsq;
        T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
        T_F_FLOAT fpair = (r6inv * (lj1(type_i,type_j)*r6inv - lj2(type_i,type_j))) * r2inv;
        fxi += dx*fpair;
        fyi += dy*fpair;
        fzi += dz*fpair;
      }
    }
    f(i,0) += fxi;
    f(i,1) += fyi;
    f(i,2) += fzi;

  }
};
#endif
#endif
