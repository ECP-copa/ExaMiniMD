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
      T = temp.compute(system);
      PE = pote.compute(system,binning,neighbor,force)/system->N;
      KE = kine.compute(system)/system->N;
      if(system->do_print) {
        double time = timer.seconds();
        printf("%i %lf %lf %lf %lf %e\n",step, T, PE, PE+KE, timer.seconds(),1.0*system->N*input->thermo_rate/(time-last_time));
        last_time = time;
      }
    }

    if (input->dumpbinaryflag)
      dump_binary(step);
    if (input->correctnessflag)
      check_correctness(step);

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

void ExaMiniMD::dump_binary(int step) {
  // On dump steps print configuration

  if(step%input->dumpbinary_rate) return;
  
  FILE* fp;
  T_INT n = system->N;
  
  if(system->do_print) {
    char* filename = new char[256];
    sprintf(filename,"%s%010d","Output/output",step);
    fp = fopen(filename,"wb");
    if (fp == NULL) {
      char str[128];
      printf("Cannot open dump file %s \n",filename);
    }
  }
    
  if(system->do_print) {
    fwrite(&n,sizeof(T_INT),1,fp);
    fwrite(&(system->id(0)),sizeof(T_INT),n,fp);
    fwrite(&(system->type(0)),sizeof(T_INT),n,fp);
    fwrite(&(system->x(0,0)),sizeof(T_X_FLOAT),3*n,fp);
    fwrite(&(system->v(0,0)),sizeof(T_V_FLOAT),3*n,fp);
    fwrite(&(system->q(0)),sizeof(T_V_FLOAT),n,fp);
  }
    
  if(system->do_print) {
    fclose(fp);
  }
}

void ExaMiniMD::check_correctness(int step) {

  if(step%input->correctness_rate) return;

  FILE* fp;
  T_INT n = system->N;
  T_INT ntmp;
  
  if(system->do_print) {
    char* filename = new char[256];
    sprintf(filename,"%s%010d","Reference/output",step);
    fp = fopen(filename,"rb");
    if (fp == NULL) {
      char str[128];
      printf("Cannot open input file %s \n",filename);
    }
  }

  if(system->do_print) {
    fread(&ntmp,sizeof(T_INT),1,fp);
    if (ntmp != n) 
      printf("Mismatch in current and reference atom counts\n");
    printf("Successfully read atom count from reference %d %d\n",n,ntmp);

    t_x xref;         // Reference Positions
    t_id   typeref;   // Reference Particle Types
    t_id   idref;     // Reference Particle IDs
    Kokkos::resize(xref,n);
    Kokkos::resize(typeref,n);
    Kokkos::resize(idref,n);

    fread(&(idref(0)),sizeof(T_INT),n,fp);
    fread(&(typeref(0)),sizeof(T_INT),n,fp); 
    fread(&(xref(0,0)),sizeof(T_X_FLOAT),3*n,fp);

    double sumdelrsq = 0.0;
    for (int i = 0; i < n; i++) {
      int ii = -1;
      if (system->id(i) != idref(i)) 
        for (int j = 0; j < n; j++) {
          if (system->id(j) == idref(i)) {
            ii = j;
            break;
          }
        }
      else
        ii = i;

      if (ii == -1)
        printf("Unable to find current id matchinf reference id %d \n",idref(i));
      else {
        double delx = system->x(ii,0)-xref(i,0);
        double dely = system->x(ii,1)-xref(i,1);
        double delz = system->x(ii,2)-xref(i,2);
        double delrsq = delx*delx + dely*dely + delz*delz;
        // if (delrsq > 0.0) 
        //   printf("delrsq = %g %g %g %g\n",delrsq,delx,dely,delz);
        // printf("%d %d %g %g %g\n",system->id(ii),system->type(ii),system->x(ii,0),system->x(ii,1),system->x(ii,2));
        // printf("%d %d %g %g %g\n",idref(i),typeref(i),xref(i,0),xref(i,1),xref(i,2));
        sumdelrsq += delrsq;
      }
    }
    printf("sumdelrsq = %g \n",sumdelrsq);
  }

  if(system->do_print) {
    fclose(fp);
  }

}

void ExaMiniMD::print_performance() {}

void ExaMiniMD::shutdown() {
  system->destroy();
}
