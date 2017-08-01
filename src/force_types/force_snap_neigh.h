/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#ifdef MODULES_OPTION_CHECK
      if( (strcmp(argv[i+1], "NEIGH_FULL") == 0) )
        force_iteration_type = FORCE_ITER_NEIGH_FULL;
#endif
#ifdef FORCE_MODULES_INSTANTIATION
    else if ((input->force_type == FORCE_SNAP) && (input->force_iteration_type == FORCE_ITER_NEIGH_FULL)) {
      force = new ForceSNAP(input->input_data.words[input->force_line],system,false);
    }
    else if ((input->force_type == FORCE_SNAP) && (input->force_iteration_type == FORCE_ITER_NEIGH_HALF)) {
      force = new ForceSNAP(input->input_data.words[input->force_line],system,true);
    }
#endif


#if !defined(MODULES_OPTION_CHECK) && !defined(FORCE_MODULES_INSTANTIATION)

#ifndef FORCE_SNAP_NEIGH_H
#define FORCE_SNAP_NEIGH_H
#include<force.h>
#include<sna.h>

class ForceSNAP : public Force {
public:
  ForceSNAP(char** args, System* system, bool half_neigh_);
  ~ForceSNAP();
  
  void init_coeff(int nargs, char** args);
  void compute(System* system, Binning* binning, Neighbor* neighbor );

  const char* name() {return "ForceSNAP";}

protected:
  System* system;

  typedef NeighListCSR<t_neigh_mem_space> t_neigh_list;
  t_neigh_list neigh_list;

  int ncoeff, ncoeffq, ncoeffall;
  typedef Kokkos::View<T_F_FLOAT**> t_bvec;
  t_bvec bvec;
  typedef Kokkos::View<T_F_FLOAT***> t_dbvec;
  t_dbvec dbvec;
  SNA sna;

  int nmax;

  // How much parallelism to use within an interaction
  int vector_length;
  // How many interactions can be run concurrently
  int concurrent_interactions;

  void allocate();
  void read_files(char *, char *);
  /*inline int equal(double* x,double* y);
  inline double dist2(double* x,double* y);
  double extra_cutoff();
  void load_balance();
  void set_sna_to_shared(int snaid,int i);
  void build_per_atom_arrays();*/

  int schedule_user;
  double schedule_time_guided;
  double schedule_time_dynamic;

  int ncalls_neigh;
  int do_load_balance;
  int ilistmask_max;
  Kokkos::View<T_INT*> ilistmast;
  int ghostinum;
  int ghostilist_max;
  Kokkos::View<T_INT*> ghostilist;
  int ghostnumneigh_max;
  Kokkos::View<T_INT*> ghostnumneigh;
  Kokkos::View<T_INT*> ghostneighs;
  Kokkos::View<T_INT*> ghostfirstneigh;
  int ghostneighs_total;
  int ghostneighs_max;

  int use_optimized;
  int use_shared_arrays;

  int i_max;
  int i_neighmax;
  int i_numpairs;
  Kokkos::View<T_INT**, Kokkos::LayoutRight> i_pairs;
  Kokkos::View<T_INT***, Kokkos::LayoutRight> i_rij;
  Kokkos::View<T_INT**, Kokkos::LayoutRight> i_inside;
  Kokkos::View<T_F_FLOAT**, Kokkos::LayoutRight> i_wj;
  Kokkos::View<T_F_FLOAT***, Kokkos::LayoutRight>i_rcutij;
  Kokkos::View<T_INT*> i_ninside;
  Kokkos::View<T_F_FLOAT****, Kokkos::LayoutRight> i_uarraytot_r, i_uarraytot_i;
  Kokkos::View<T_F_FLOAT******, Kokkos::LayoutRight> i_zarray_r, i_zarray_i;

#ifdef TIMING_INFO
  //  timespec starttime, endtime;
  double timers[4];
#endif

  double rcutmax;               // max cutoff for all elements
  int nelements;                // # of unique elements
  char **elements;              // names of unique elements
  Kokkos::View<T_F_FLOAT*> radelem;              // element radii
  Kokkos::View<T_F_FLOAT*> wjelem;               // elements weights
  Kokkos::View<T_F_FLOAT**, Kokkos::LayoutRight> coeffelem;           // element bispectrum coefficients
  Kokkos::View<T_INT*> map;                     // mapping from atom types to elements
  int twojmax, diagonalstyle, switchflag, bzeroflag, quadraticflag;
  double rcutfac, rfac0, rmin0, wj1, wj2;
  int rcutfacflag, twojmaxflag; // flags for required parameters
  typedef Kokkos::View<T_F_FLOAT**> t_fparams;
  t_fparams cutsq;
  typedef Kokkos::View<const T_F_FLOAT**,
      Kokkos::MemoryTraits<Kokkos::RandomAccess>> t_fparams_rnd;
  t_fparams_rnd rnd_cutsq;


  t_x x;
  t_f_atomic f;
  t_type type;

public:

  KOKKOS_INLINE_FUNCTION
  void operator() (const Kokkos::TeamPolicy<>::member_type& team) const {
    const int i = team.league_rank();
    SNA my_sna(sna,team);
    const double x_i = x(i,0);
    const double y_i = x(i,1);
    const double z_i = x(i,2);
    const int type_i = type[i];
    const int elem_i = map[type_i];
    const double radi = radelem[elem_i];

    typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);

    const int num_neighs = neighs_i.get_num_neighs();

    // rij[][3] = displacements between atom I and those neighbors
    // inside = indices of neighbors of I within cutoff
    // wj = weights for neighbors of I within cutoff
    // rcutij = cutoffs for neighbors of I within cutoff
    // note Rij sign convention => dU/dRij = dU/dRj = -dU/dRi

    int ninside = 0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,num_neighs),
        [&] (const int jj, int& count) {
      Kokkos::single(Kokkos::PerThread(team), [&] (){
        T_INT j = neighs_i(jj);
        const T_F_FLOAT dx = x(j,0) - x_i;
        const T_F_FLOAT dy = x(j,1) - y_i;
        const T_F_FLOAT dz = x(j,2) - z_i;

        const int type_j = type(j);
        const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;
        const int elem_j = map[type_j];

        if( rsq < rnd_cutsq(type_i,type_j) )
         count++;
      });
    },ninside);

    if(team.team_rank() == 0)
    Kokkos::parallel_scan(Kokkos::ThreadVectorRange(team,num_neighs),
        [&] (const int jj, int& offset, bool final){
    //for (int jj = 0; jj < num_neighs; jj++) {
      T_INT j = neighs_i(jj);
      const T_F_FLOAT dx = x(j,0) - x_i;
      const T_F_FLOAT dy = x(j,1) - y_i;
      const T_F_FLOAT dz = x(j,2) - z_i;

      const int type_j = type(j);
      const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;
      const int elem_j = map[type_j];

      if( rsq < rnd_cutsq(type_i,type_j) ) {
        if(final) {
          my_sna.rij(offset,0) = dx;
          my_sna.rij(offset,1) = dy;
          my_sna.rij(offset,2) = dz;
          my_sna.inside[offset] = j;
          my_sna.wj[offset] = wjelem[elem_j];
          my_sna.rcutij[offset] = (radi + radelem[elem_j])*rcutfac;
        }
        offset++;
      }
    });

    team.team_barrier();
    // compute Ui, Zi, and Bi for atom I
    my_sna.compute_ui(team,ninside);
    team.team_barrier();
    my_sna.compute_zi(team);
    team.team_barrier();

    // for neighbors of I within cutoff:
    // compute dUi/drj and dBi/drj
    // Fij = dEi/dRj = -dEi/dRi => add to Fi, subtract from Fj

    Kokkos::View<double*,Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      coeffi(coeffelem,elem_i,Kokkos::ALL);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,ninside),
        [&] (const int jj) {
    //for (int jj = 0; jj < ninside; jj++) {
      int j = my_sna.inside[jj];
      my_sna.compute_duidrj(team,&my_sna.rij(jj,0),
                             my_sna.wj[jj],my_sna.rcutij[jj]);

      my_sna.compute_dbidrj(team);
      my_sna.copy_dbi2dbvec(team);


      Kokkos::single(Kokkos::PerThread(team), [&] (){
      T_F_FLOAT fij[3];

      fij[0] = 0.0;
      fij[1] = 0.0;
      fij[2] = 0.0;

      // linear contributions

      for (int k = 1; k <= ncoeff; k++) {
        double bgb = coeffi[k];
        fij[0] += bgb*my_sna.dbvec(k-1,0);
        fij[1] += bgb*my_sna.dbvec(k-1,1);
        fij[2] += bgb*my_sna.dbvec(k-1,2);
      }

      const double dx = my_sna.rij(jj,0);
      const double dy = my_sna.rij(jj,1);
      const double dz = my_sna.rij(jj,2);
      const double fdivr = -1.5e6/pow(dx*dx + dy*dy + dz*dz,7.0);
      fij[0] += dx*fdivr;
      fij[1] += dy*fdivr;
      fij[2] += dz*fdivr;

      //OK
      //printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf SNAP-COMPARE: FIJ\n"
      //    ,x(i,0),x(i,1),x(i,2),x(j,0),x(j,1),x(j,2),fij[0],fij[1],fij[2] );
      f(i,0) += fij[0];
      f(i,1) += fij[1];
      f(i,2) += fij[2];
      f(j,0) -= fij[0];
      f(j,1) -= fij[1];
      f(j,2) -= fij[2];
      });
    });
  }
};


#endif
#endif
