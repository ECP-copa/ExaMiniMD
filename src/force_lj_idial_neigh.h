#ifdef MODULES_OPTION_CHECK
      if( (strcmp(argv[i+1], "NEIGH_FULL") == 0) )
        force_iteration_type = FORCE_ITER_NEIGH_FULL;
      if( (strcmp(argv[i+1], "NEIGH_HALF") == 0) ) {
        force_iteration_type = FORCE_ITER_NEIGH_HALF;
      }
#endif
#ifdef MODULES_INSTANTIATION
    else if ((input->force_type == FORCE_LJ_IDIAL) && (input->force_iteration_type == FORCE_ITER_NEIGH_FULL)) {
      force = new ForceLJIDialNeigh(input->input_data.words[input->force_line],system,false);
    }
    else if ((input->force_type == FORCE_LJ_IDIAL) && (input->force_iteration_type == FORCE_ITER_NEIGH_HALF)) {
      force = new ForceLJIDialNeigh(input->input_data.words[input->force_line],system,true);
    }
#endif


#if !defined(MODULES_OPTION_CHECK) && !defined(MODULES_INSTANTIATION)

#ifndef FORCE_LJ_IDIAL_NEIGH_H
#define FORCE_LJ_IDIAL_NEIGH_H
#include<force.h>

class ForceLJIDialNeigh: public Force {
private:
  int N_local;
  t_x_const_rnd x;
  t_f f;
  t_f_atomic f_a;
  t_id id;
  t_type_const_rnd type;
  Binning::t_bincount bin_count;
  Binning::t_binoffsets bin_offsets;
  T_INT nbinx,nbiny,nbinz,nhalo;
  int step;

  typedef Kokkos::View<T_F_FLOAT**> t_fparams;
  typedef Kokkos::View<const T_F_FLOAT**,
      Kokkos::MemoryTraits<Kokkos::RandomAccess>> t_fparams_rnd;
  t_fparams lj1,lj2,cutsq,intensity;
  t_fparams_rnd rnd_lj1,rnd_lj2,rnd_cutsq,rnd_intensity;
  typedef NeighListCSR<t_neigh_mem_space> t_neigh_list;

  t_neigh_list neigh_list;

public:
  struct TagFullNeigh {};
  struct TagHalfNeigh {};

  typedef Kokkos::RangePolicy<TagFullNeigh,Kokkos::IndexType<T_INT> > t_policy_full_neigh;
  typedef Kokkos::RangePolicy<TagHalfNeigh,Kokkos::IndexType<T_INT> > t_policy_half_neigh;

  ForceLJIDialNeigh (char** args, System* system, bool half_neigh_);

  void init_coeff(int nargs, char** args);

  void compute(System* system, Binning* binning, Neighbor* neighbor );

  KOKKOS_INLINE_FUNCTION
  void operator() (TagFullNeigh, const T_INT& i) const {
    const T_F_FLOAT x_i = x(i,0);
    const T_F_FLOAT y_i = x(i,1);
    const T_F_FLOAT z_i = x(i,2);
    const int type_i = type(i);
    const int type_j = type(j);

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
				for(int repeat = 0; repeat < intensity(type_i,type_j); repeat++) {
					T_F_FLOAT r2inv = 1.0/rsq;
        	T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
        	T_F_FLOAT fpair = (r6inv * (rnd_lj1(type_i,type_j)*r6inv - rnd_lj2(type_i,type_j))) * r2inv;
        	fxi += dx*fpair;
        	fyi += dy*fpair;
        	fzi += dz*fpair;
				}
      }
    }
    f(i,0) += fxi/intensity(type_i,type_j);
    f(i,1) += fyi/intensity(type_i,type_j);
    f(i,2) += fzi/intensity(type_i,type_j);

  }

  KOKKOS_INLINE_FUNCTION
  void operator() (TagHalfNeigh, const T_INT& i) const {
    const T_F_FLOAT x_i = x(i,0);
    const T_F_FLOAT y_i = x(i,1);
    const T_F_FLOAT z_i = x(i,2);
    const int type_i = type(i);
    const int type_j = type(j);

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
				for(int repeat = 0; repeat < intensity(type_i,type_j); repeat++) {
					T_F_FLOAT r2inv = 1.0/rsq;
        	T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
        	T_F_FLOAT fpair = (r6inv * (rnd_lj1(type_i,type_j)*r6inv - rnd_lj2(type_i,type_j))) * r2inv;
        	fxi += dx*fpair;
        	fyi += dy*fpair;
        	fzi += dz*fpair;
        	if(j<N_local) {
        	  f_a(j,0) -= dx*fpair/intensity(type_i,type_j);
        	  f_a(j,1) -= dy*fpair/intensity(type_i,type_j);
        	  f_a(j,2) -= dz*fpair/intensity(type_i,type_j);
        	}
				}
      }
    }
    f_a(i,0) += fxi/intensity(type_i,type_j);
    f_a(i,1) += fyi/intensity(type_i,type_j);
    f_a(i,2) += fzi/intensity(type_i,type_j);

  }

  const char* name();
};
#endif
#endif
