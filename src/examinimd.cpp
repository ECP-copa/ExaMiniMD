#include <examinimd.h>
#include <property_temperature.h>

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
    printf("Using: %s %s\n",neighbor->name(),comm->name());
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

  Kokkos::Timer timer;
  // Timestep Loop
  for(int step = 1; step <= nsteps; step++ ) {
    
    // Do first part of the verlet time step integration 
    integrator->initial_integrate();

    if(step%input->comm_exchange_rate==0 && step >0) {
      // Exchange particles
      comm->exchange(); 

      // Sort particles
      binning->create_binning(neigh_cutoff,neigh_cutoff,neigh_cutoff,1,true,false,true);

      // Exchange Halo
      comm->exchange_halo();
      
      // Create binning for neighborlist construction
      binning->create_binning(neigh_cutoff,neigh_cutoff,neigh_cutoff,1,true,true,false);

      // Compute Neighbor List if necessary
      if(neighbor)
        neighbor->create_neigh_list(system,binning,force->half_neigh,false);
    } else {
      // Exchange Halo
      comm->update_halo();
    }

    // Zero out forces 
    Kokkos::deep_copy(system->f,0.0);
   
    // Compute Short Range Force
    force->compute(system,binning,neighbor);

    // This is where Bonds, Angles and KSpace should go eventually 
    
    // Do second part of the verlet time step integration 
    integrator->final_integrate();

    // On output steps print output
    if(step%input->thermo_rate==0) {
      Temperature temp(comm);
      double T = temp.compute(system);
      if(system->do_print)
        printf("%i Temperature: %lf\n",step, T);
    }
  }

  Temperature temp(comm);
  double T = temp.compute(system);
  if(system->do_print)
    printf("Finished: T %lf Time: %lf\n",T,timer.seconds());
}

void ExaMiniMD::check_correctness() {}

void ExaMiniMD::print_performance() {}

void ExaMiniMD::shutdown() {
  system->destroy();
}
