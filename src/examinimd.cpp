#include <examinimd.h>
#include <property_temperature.h>
#include <property_kine.h>
#include <property_pote.h>

ExaMiniMD::ExaMiniMD() {
  // First we need to create the System data structures
  // They are used by input
  system = new System();
  system->init();

  // Create the Input System, no modules for that,
  // so we can init it in constructor
  input = new Input(system);

  neighbor = NULL;
}

void ExaMiniMD::init(int argc, char* argv[]) {

  if(system->do_print)
    Kokkos::DefaultExecutionSpace::print_configuration(std::cout);

  // Lets parse the command line arguments
  input->read_command_line_args(argc,argv);

  // Read input file 
  input->read_file();

  // Now we know which integrator type to use
  if(input->integrator_type == INTEGRATOR_NVE)
    integrator = new IntegratorNVE(system);
  
  // Fill some binning
  if(input->binning_type == BINNING_KKSORT)
    binning = new BinningKKSort(system);

  // Create Force Type
  if(false) {}
#define MODULES_INSTANTIATION
#include<modules_force.h>
#undef MODULES_INSTANTIATION
  else if(system->do_print) {
    printf("Error: Invalid ForceType\n");
    exit(0);
  }
  for(int line = 0; line < input->force_coeff_lines.dimension_0(); line++) {
    force->init_coeff(6,input->input_data.words[input->force_coeff_lines(line)]);
  }

  // Create Neighbor Instance
  if (false) {}
#define MODULES_INSTANTIATION
#include<modules_neighbor.h>
#undef MODULES_INSTANTIATION
  else if(system->do_print)
    printf("Error: Invalid NeighborType\n");

  // Create Communication Submodule
  if (false) {}
#define MODULES_INSTANTIATION
#include<modules_comm.h>
#undef MODULES_INSTANTIATION
  else if(system->do_print)
    printf("Error: Invalid CommType\n");

  // system->print_particles();
  if(system->do_print) {
    printf("Using: %s %s %s %s\n",force->name(),neighbor->name(),comm->name(),binning->name());
  }

  // Ok lets go ahead and create the particles if that didn't happen yet
  if(system->N == 0)
    input->create_lattice(comm);

  comm->exchange(); 

  // Sort particles
  T_F_FLOAT neigh_cutoff = input->force_cutoff + input->neighbor_skin;
  binning->create_binning(neigh_cutoff,neigh_cutoff,neigh_cutoff,1,true,false,true);

  // Set up particles
  comm->exchange_halo();

  // Create binning for neighborlist construction
  binning->create_binning(neigh_cutoff,neigh_cutoff,neigh_cutoff,1,true,true,false);

  // Compute NeighList
  if(neighbor)
    neighbor->create_neigh_list(system,binning,force->half_neigh,false);

  // Compute initial forces
  force->compute(system,binning,neighbor);

}

void ExaMiniMD::run(int nsteps) {
  T_F_FLOAT neigh_cutoff = input->force_cutoff + input->neighbor_skin;
  Temperature temp(comm);
  PotE pote(comm);
  KinE kine(comm);
  double T = temp.compute(system);
  double PE = pote.compute(system,binning,neighbor,force)/system->N;
  double KE = kine.compute(system)/system->N;
  if(system->do_print) {
    printf("\n");
    printf("#Timestep Temperature PotE ETot Time Atomsteps/s\n");
    printf("%i %lf %lf %lf %lf %e\n",0,T,PE,PE+KE,0.0,0.0);
  }

  double force_time = 0;
  double comm_time  = 0;
  double neigh_time = 0;
  double other_time = 0;

  double last_time;
  Kokkos::Timer timer,force_timer,comm_timer,neigh_timer,other_timer;

  // Timestep Loop
  for(int step = 1; step <= nsteps; step++ ) {
    
    // Do first part of the verlet time step integration 
    other_timer.reset();
    integrator->initial_integrate();
    other_time += other_timer.seconds();

    if(step%input->comm_exchange_rate==0 && step >0) {
      // Exchange particles
      comm_timer.reset();
      comm->exchange(); 
      comm_time += comm_timer.seconds();

      // Sort particles
      other_timer.reset();
      binning->create_binning(neigh_cutoff,neigh_cutoff,neigh_cutoff,1,true,false,true);
      other_time += other_timer.seconds();

      // Exchange Halo
      comm_timer.reset();
      comm->exchange_halo();
      comm_time += comm_timer.seconds();
      
      // Create binning for neighborlist construction
      neigh_timer.reset();
      binning->create_binning(neigh_cutoff,neigh_cutoff,neigh_cutoff,1,true,true,false);

      // Compute Neighbor List if necessary
      if(neighbor)
        neighbor->create_neigh_list(system,binning,force->half_neigh,false);
      neigh_time += neigh_timer.seconds();
    } else {
      // Exchange Halo
      comm_timer.reset();
      comm->update_halo();
      comm_time += comm_timer.seconds();
    }

    // Zero out forces 
    force_timer.reset();
    Kokkos::deep_copy(system->f,0.0);
   
    // Compute Short Range Force
    force->compute(system,binning,neighbor);
    force_time += force_timer.seconds();

    // This is where Bonds, Angles and KSpace should go eventually 
    
    // Do second part of the verlet time step integration 
    other_timer.reset();
    integrator->final_integrate();

    // On output steps print output
    if(step%input->thermo_rate==0) {
      double T = temp.compute(system);
      double PE = pote.compute(system,binning,neighbor,force)/system->N;
      double KE = kine.compute(system)/system->N;
      if(system->do_print) {
        double time = timer.seconds();
        printf("%i %lf %lf %lf %lf %e\n",step, T, PE, PE+KE, timer.seconds(),1.0*system->N*input->thermo_rate/(time-last_time));
        last_time = time;
      }
    }
    other_time += other_timer.seconds();
  }

  double time = timer.seconds();
  T = temp.compute(system);
  PE = pote.compute(system,binning,neighbor,force)/system->N;
  KE = kine.compute(system)/system->N;

  if(system->do_print) {
    printf("\n");
    printf("#Procs Particles | Time T_Force T_Neigh T_Comm T_Other | Steps/s Atomsteps/s Atomsteps/(proc*s)\n");
    printf("%i %i | %lf %lf %lf %lf %lf | %lf %e %e PERFORMANCE\n",comm->num_processes(),system->N,time,
      force_time,neigh_time,comm_time,other_time,
      1.0*nsteps/time,1.0*system->N*nsteps/time,1.0*system->N*nsteps/time/comm->num_processes());
  }
}

void ExaMiniMD::check_correctness() {}

void ExaMiniMD::print_performance() {}

void ExaMiniMD::shutdown() {
  system->destroy();
}
