#include<system.h>
#ifdef EXAMINIMD_ENABLE_MPI
#include<mpi.h>
#endif
System::System() {
  N = 0;
  N_max = 0;
  N_local = 0;
  N_ghost = 0;
  ntypes = 1;
  x = t_x();
  v = t_v();
  f = t_f();
  id = t_id();
  type = t_type();
  q = t_q();
  mass = t_mass();
  domain_x = domain_y = domain_z = 0.0;
  sub_domain_x = sub_domain_y = sub_domain_z = 0.0;
  sub_domain_hi_x = sub_domain_hi_y = sub_domain_hi_z = 0.0;
  sub_domain_lo_x = sub_domain_lo_y = sub_domain_lo_z = 0.0;
  mvv2e = boltz = dt = 0.0;
#ifdef EXAMINIMD_ENABLE_MPI
  int proc_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  do_print = proc_rank == 0;
#else
  do_print = true;
#endif
}

void System::init() {
  x = t_x("System::x",N_max);
  v = t_v("System::v",N_max);
  f = t_f("System::f",N_max);
  id = t_id("System::id",N_max);
  type = t_type("System::type",N_max);
  q = t_q("System::q",N_max);
  mass = t_mass("System::mass",ntypes);
}

void System::destroy() {
  N_max = 0;
  N_local = 0;
  N_ghost = 0;
  ntypes = 1;
  x = t_x();
  v = t_v();
  f = t_f();
  id = t_id();
  type = t_type();
  q = t_q();
  mass = t_mass();
}

void System::grow(T_INT N_new) {
  if(N_new > N_max) {
    N_max = N_new; // Number of global Particles

    Kokkos::resize(x,N_max);      // Positions
    Kokkos::resize(v,N_max);      // Velocities
    Kokkos::resize(f_r,N_max);      // Forces
    f = f_r.subview();

    Kokkos::resize(id,N_max);     // Id

    Kokkos::resize(type,N_max);   // Particle Type

    Kokkos::resize(q,N_max);      // Charge
  }
}

void System::print_particles() {
  printf("Print all particles: \n");
  printf("  Owned: %d\n",N_local);
  for(T_INT i=0;i<N_local;i++) {
    printf("    %d %lf %lf %lf | %lf %lf %lf | %lf %lf %lf | %d %e\n",i,
        double(x(i,0)),double(x(i,1)),double(x(i,2)),
        double(v(i,0)),double(v(i,1)),double(v(i,2)),
        double(f(i,0)),double(f(i,1)),double(f(i,2)),
        type(i),q(i)
        );
  }

  printf("  Ghost: %d\n",N_ghost);
  for(T_INT i=N_local;i<N_local+N_ghost;i++) {
    printf("    %d %lf %lf %lf | %lf %lf %lf | %lf %lf %lf | %d %e\n",i,
        double(x(i,0)),double(x(i,1)),double(x(i,2)),
        double(v(i,0)),double(v(i,1)),double(v(i,2)),
        double(f(i,0)),double(f(i,1)),double(f(i,2)),
        type(i),q(i)
        );
  }

}
