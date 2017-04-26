#ifdef EXAMINIMD_ENABLE_MPI
#include<comm_mpi.h>

CommMPI::CommMPI(System* s, T_X_FLOAT comm_depth):Comm(s,comm_depth) {
  pack_count = Kokkos::View<int>("CommMPI::pack_count");
  pack_buffer = Kokkos::View<Particle*>("CommMPI::pack_buffer",200);
  unpack_buffer = Kokkos::View<Particle*>("CommMPI::pack_buffer",200);
  pack_indicies_all = Kokkos::View<T_INT**,Kokkos::LayoutRight>("CommMPI::pack_indicies_all",6,200);
  exchange_dest_list = Kokkos::View<T_INT*,Kokkos::LayoutRight >("CommMPI::exchange_dest_list",200);
}

void CommMPI::init() {};

void CommMPI::create_domain_decomposition() {

  //printf("Domain b: %p %lf %lf %lf\n",system,system->domain_x ,system->domain_y ,system->domain_z);

  MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  int ipx = 1;

  double area_xy = system->domain_x * system->domain_y;
  double area_xz = system->domain_x * system->domain_z;
  double area_yz = system->domain_y * system->domain_z;

  double smallest_surface = 2.0 * (area_xy + area_xz + area_yz);

  while(ipx <= proc_size) {
    if(proc_size % ipx == 0) {
      int nremain = proc_size / ipx;
      int ipy = 1;

      while(ipy <= nremain) {
        if(nremain % ipy == 0) {
          int ipz = nremain / ipy;
          double surface = area_xy / ipx / ipy + area_xz / ipx / ipz + area_yz / ipy / ipz;

          if(surface < smallest_surface) {
            smallest_surface = surface;
            proc_grid[0] = ipx;
            proc_grid[1] = ipy;
            proc_grid[2] = ipz;
          }
        }

        ipy++;
      }
    }

    ipx++;
  }
  proc_pos[2] = proc_rank / (proc_grid[0] * proc_grid[1]);
  proc_pos[1] = (proc_rank % (proc_grid[0] * proc_grid[1])) / proc_grid[0];
  proc_pos[0] = proc_rank % proc_grid[0];

  if(proc_grid[0]>1) {
    proc_neighbors_send[1] = ( proc_pos[0] > 0 )                ? proc_rank - 1 :
                                                                proc_rank + (proc_grid[0]-1);
    proc_neighbors_send[0] = ( proc_pos[0] < (proc_grid[0]-1) ) ? proc_rank + 1 :
                                                                proc_rank - (proc_grid[0]-1);

  } else {
    proc_neighbors_send[0] = -1;
    proc_neighbors_send[1] = -1;
  }

  if(proc_grid[1]>1) {
    proc_neighbors_send[3] = ( proc_pos[1] > 0 )                ? proc_rank - proc_grid[0] :
                                                                proc_rank + proc_grid[0]*(proc_grid[1]-1);
    proc_neighbors_send[2] = ( proc_pos[1] < (proc_grid[1]-1) ) ? proc_rank + proc_grid[0] :
                                                                proc_rank - proc_grid[0]*(proc_grid[1]-1);
  } else {
    proc_neighbors_send[2] = -1;
    proc_neighbors_send[3] = -1;
  }

  if(proc_grid[2]>1) {
    proc_neighbors_send[5] = ( proc_pos[2] > 0 )                ? proc_rank - proc_grid[0]*proc_grid[1] :
                                                                proc_rank + proc_grid[0]*proc_grid[1]*(proc_grid[2]-1);
    proc_neighbors_send[4] = ( proc_pos[2] < (proc_grid[2]-1) ) ? proc_rank + proc_grid[0]*proc_grid[1] :
                                                                proc_rank - proc_grid[0]*proc_grid[1]*(proc_grid[2]-1);
  } else {
    proc_neighbors_send[4] = -1;
    proc_neighbors_send[5] = -1;
  }

  proc_neighbors_recv[0] = proc_neighbors_send[1];
  proc_neighbors_recv[1] = proc_neighbors_send[0];
  proc_neighbors_recv[2] = proc_neighbors_send[3];
  proc_neighbors_recv[3] = proc_neighbors_send[2];
  proc_neighbors_recv[4] = proc_neighbors_send[5];
  proc_neighbors_recv[5] = proc_neighbors_send[4];

  system->sub_domain_x = system->domain_x / proc_grid[0];
  system->sub_domain_y = system->domain_y / proc_grid[1];
  system->sub_domain_z = system->domain_z / proc_grid[2];
  system->sub_domain_lo_x = proc_pos[0] * system->sub_domain_x;
  system->sub_domain_lo_y = proc_pos[1] * system->sub_domain_y;
  system->sub_domain_lo_z = proc_pos[2] * system->sub_domain_z;
  system->sub_domain_hi_x = ( proc_pos[0] + 1 ) * system->sub_domain_x;
  system->sub_domain_hi_y = ( proc_pos[1] + 1 ) * system->sub_domain_y;
  system->sub_domain_hi_z = ( proc_pos[2] + 1 ) * system->sub_domain_z;

  /*printf("MyRank: %i MyPos: %i %i %i Grid: %i %i %i Neighbors: %i %i , %i %i , %i %i\n",proc_rank,proc_pos[0],
      proc_pos[1],proc_pos[2],proc_grid[0],proc_grid[1],proc_grid[2],
      proc_neighbors_send[0],proc_neighbors_send[1],
      proc_neighbors_send[2],proc_neighbors_send[3],
      proc_neighbors_send[4],proc_neighbors_send[5]);*/
}


void CommMPI::scan_int(T_INT* vals, T_INT count) {
  if(std::is_same<T_INT,int>::value) {
    MPI_Scan(vals,vals,count,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  }
}

void CommMPI::reduce_int(T_INT* vals, T_INT count) {
  if(std::is_same<T_INT,int>::value) {
    MPI_Allreduce(vals,vals,count,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  }
}

void CommMPI::reduce_float(T_FLOAT* vals, T_INT count) {
  if(std::is_same<T_FLOAT,double>::value) {
    MPI_Allreduce(vals,vals,count,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  }
}

void CommMPI::exchange() {
  s = *system;
  N_local = system->N_local;
  N_ghost = 0;
  //printf("System A: %i %lf %lf %lf %i\n",s.N_local,s.x(21,0),s.x(21,1),s.x(21,2),s.type(21));
  Kokkos::parallel_for("CommMPI::exchange_self",
            Kokkos::RangePolicy<TagExchangeSelf, Kokkos::IndexType<T_INT> >(0,N_local), *this);

  T_INT N_total_recv = 0;
  T_INT N_total_send = 0;

  //printf("System B: %i %lf %lf %lf %i\n",s.N_local,s.x(21,0),s.x(21,1),s.x(21,2),s.type(21));
  for(phase = 0; phase < 6; phase ++) {
    proc_num_send[phase] = 0;
    proc_num_recv[phase] = 0;
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());

    T_INT count = 0;
    Kokkos::deep_copy(pack_count,0);

    if(proc_grid[phase/2]>1) {
      Kokkos::parallel_for("CommMPI::exchange_pack",
                Kokkos::RangePolicy<TagExchangePack, Kokkos::IndexType<T_INT> >(0,N_local+N_ghost),
                *this);

      Kokkos::deep_copy(count,pack_count);
      if(count > pack_indicies.extent(0)) {
        Kokkos::realloc(pack_buffer,count*1.1);
        Kokkos::realloc(pack_indicies_all,6,count*1.1);
        pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
        Kokkos::deep_copy(pack_count,0);
        Kokkos::parallel_for("CommMPI::exchange_pack",
                  Kokkos::RangePolicy<TagExchangePack, Kokkos::IndexType<T_INT> >(0,N_local+N_ghost),
                  *this);
      }
      proc_num_send[phase] = count;
      if(pack_buffer.extent(0)<count)
        pack_buffer = Kokkos::View<Particle*>("Comm::pack_buffer",count);
      MPI_Request request;
      MPI_Irecv(&proc_num_recv[phase],1,MPI_INT, proc_neighbors_recv[phase],100001,MPI_COMM_WORLD,&request);
      MPI_Send(&proc_num_send[phase],1,MPI_INT, proc_neighbors_send[phase],100001,MPI_COMM_WORLD);
      MPI_Status status;
      MPI_Wait(&request,&status);
      count = proc_num_recv[phase];
      //printf("Recv Count: %i\n",count);
      if(unpack_buffer.extent(0)<count) {
        unpack_buffer = Kokkos::View<Particle*>("Comm::unpack_buffer",count);
      }
      if(proc_num_recv[phase]>0)
        MPI_Irecv(unpack_buffer.data(),proc_num_recv[phase]*sizeof(Particle)/sizeof(int),MPI_INT, proc_neighbors_recv[phase],100002,MPI_COMM_WORLD,&request);
      if(proc_num_send[phase]>0)
        MPI_Send (pack_buffer.data(),proc_num_send[phase]*sizeof(Particle)/sizeof(int),MPI_INT, proc_neighbors_send[phase],100002,MPI_COMM_WORLD);
      system->grow(N_local + N_ghost + count);
      s = *system;
      if(proc_num_recv[phase]>0)
        MPI_Wait(&request,&status);
      Kokkos::parallel_for("CommMPI::exchange_unpack",
                Kokkos::RangePolicy<TagUnpack, Kokkos::IndexType<T_INT> >(0,proc_num_recv[phase]),
                *this);

    }

    N_ghost += count;
    N_total_recv += proc_num_recv[phase];
    N_total_send += proc_num_send[phase];
  }
  T_INT N_local_start = N_local;
  T_INT N_exchange = N_ghost;
  //printf("System C: %i %lf %lf %lf %i\n",s.N_local,s.x(21,0),s.x(21,1),s.x(21,2),s.type(21));

  N_local = N_local + N_total_recv - N_total_send;
  N_ghost = N_local_start + N_exchange - N_local;

  if(exchange_dest_list.extent(0)<N_ghost)
    Kokkos::realloc(exchange_dest_list,N_ghost);

  //printf("ExchangeDestList: %i %i %i %i %i\n",exchange_dest_list.extent(0),N_ghost,N_total_recv,N_total_send,N_local);
  //printf("System: %i %i\n",s.N_local,system->N_local);
  Kokkos::parallel_scan("CommMPI::exchange_create_dest_list",
            Kokkos::RangePolicy<TagExchangeCreateDestList, Kokkos::IndexType<T_INT> >(0,N_local),
            *this);
  //printf("System D: %i %lf %lf %lf %i\n",s.N_local,s.x(21,0),s.x(21,1),s.x(21,2),s.type(21));
  Kokkos::parallel_scan("CommMPI::exchange_compact",
            Kokkos::RangePolicy<TagExchangeCompact, Kokkos::IndexType<T_INT> >(0,N_ghost),
            *this);
  //printf("System E: %i %lf %lf %lf %i\n",s.N_local,s.x(21,0),s.x(21,1),s.x(21,2),s.type(21));

  system->N_local = N_local;
  system->N_ghost = 0;
  /*for(int i=0;i<N_local;i++)
    if(s.type(i)<0) printf("Huch: %i %i\n",i,N_local);*/
};

void CommMPI::exchange_halo() {

  N_local = system->N_local;
  N_ghost = 0;

  s = *system;

  for(phase = 0; phase < 6; phase ++) {
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());

    T_INT count = 0;
    Kokkos::deep_copy(pack_count,0);

    if(proc_grid[phase/2]>1) {
      Kokkos::parallel_for("CommMPI::halo_exchange_pack",
                Kokkos::RangePolicy<TagHaloPack, Kokkos::IndexType<T_INT> >(0,N_local+N_ghost - ( (phase%2==1) ? proc_num_recv[phase-1]:0)),
                *this);

      Kokkos::deep_copy(count,pack_count);
      if(count > pack_indicies.extent(0)) {
        Kokkos::realloc(pack_buffer,count*1.1);
        Kokkos::resize(pack_indicies_all,6,count*1.1);
        pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
        Kokkos::deep_copy(pack_count,0);
        Kokkos::parallel_for("CommMPI::halo_exchange_pack",
                  Kokkos::RangePolicy<TagHaloPack, Kokkos::IndexType<T_INT> >(0,N_local+N_ghost- ( (phase%2==1) ? proc_num_recv[phase-1]:0)),
                  *this);
      }
      proc_num_send[phase] = count;
      if(pack_buffer.extent(0)<count)
        pack_buffer = Kokkos::View<Particle*>("Comm::pack_buffer",count);
      MPI_Request request;
      MPI_Irecv(&proc_num_recv[phase],1,MPI_INT, proc_neighbors_recv[phase],100001,MPI_COMM_WORLD,&request);
      MPI_Send(&proc_num_send[phase],1,MPI_INT, proc_neighbors_send[phase],100001,MPI_COMM_WORLD);
      MPI_Status status;
      MPI_Wait(&request,&status);
      count = proc_num_recv[phase];
      if(unpack_buffer.extent(0)<count) {
        unpack_buffer = Kokkos::View<Particle*>("Comm::unpack_buffer",count);
      }
      MPI_Irecv(unpack_buffer.data(),proc_num_recv[phase]*sizeof(Particle)/sizeof(int),MPI_INT, proc_neighbors_recv[phase],100002,MPI_COMM_WORLD,&request);
      MPI_Send (pack_buffer.data(),proc_num_send[phase]*sizeof(Particle)/sizeof(int),MPI_INT, proc_neighbors_send[phase],100002,MPI_COMM_WORLD);
      system->grow(N_local + N_ghost + count);
      s = *system;
      MPI_Wait(&request,&status);
      Kokkos::parallel_for("CommMPI::halo_exchange_unpack",
                Kokkos::RangePolicy<TagUnpack, Kokkos::IndexType<T_INT> >(0,proc_num_recv[phase]),
                *this);

    } else {
      T_INT nparticles = N_local + N_ghost - ( (phase%2==1) ? proc_num_recv[phase-1]:0 );
      Kokkos::parallel_for("CommMPI::halo_exchange_self",
                Kokkos::RangePolicy<TagHaloSelf, Kokkos::IndexType<T_INT> >(0,nparticles),
                *this);
      Kokkos::deep_copy(count,pack_count);
      bool redo = false;
      if(N_local+N_ghost+count>s.x.extent(0)) {
        system->grow(N_local + N_ghost + count);
        s = *system;
        redo = true;
      }
      if(count > pack_indicies.extent(0)) {
        Kokkos::realloc(pack_buffer,count*1.1);
        Kokkos::resize(pack_indicies_all,6,count*1.1);
        pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
        redo = true;
      }
      if(redo) {
        Kokkos::deep_copy(pack_count,0);
        Kokkos::parallel_for("CommMPI::halo_exchange_self",
                  Kokkos::RangePolicy<TagHaloSelf, Kokkos::IndexType<T_INT> >(0,nparticles),
                  *this);
      }
      //printf("ExchangeHalo: %i %i %i %i\n",phase,count,pack_indicies_all.extent(1),pack_indicies.extent(0));
      proc_num_send[phase] = count;
      proc_num_recv[phase] = count;
    }

    N_ghost += count;
  }
  static int step = 0;
  step++;

  system->N_ghost = N_ghost;

};

void CommMPI::update_halo() {
  N_ghost = 0;
  s=*system;
  for(phase = 0; phase<6; phase++) {
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
    if(proc_grid[phase/2]>1) {  
      
      Kokkos::parallel_for("CommMPI::halo_update_pack",
         Kokkos::RangePolicy<TagHaloUpdatePack, Kokkos::IndexType<T_INT> >(0,proc_num_send[phase]),
         *this);
      MPI_Request request;
      MPI_Status status;
      MPI_Irecv(unpack_buffer.data(),proc_num_recv[phase]*sizeof(Particle)/sizeof(int),MPI_INT, proc_neighbors_recv[phase],100002,MPI_COMM_WORLD,&request);
      MPI_Send (pack_buffer.data(),proc_num_send[phase]*sizeof(Particle)/sizeof(int),MPI_INT, proc_neighbors_send[phase],100002,MPI_COMM_WORLD);
      s = *system;
      MPI_Wait(&request,&status);
      Kokkos::parallel_for("CommMPI::halo_update_unpack",
                Kokkos::RangePolicy<TagUnpack, Kokkos::IndexType<T_INT> >(0,proc_num_recv[phase]),
                *this);

    } else {
      //printf("HaloUpdateCopy: %i %i %i\n",phase,proc_num_send[phase],pack_indicies.extent(0));
      Kokkos::parallel_for("CommMPI::halo_update_self",
        Kokkos::RangePolicy<TagHaloUpdateSelf, Kokkos::IndexType<T_INT> >(0,proc_num_send[phase]),
        *this);
    }
    N_ghost += proc_num_recv[phase];
  }
};

const char* CommMPI::name() { return "CommMPI"; }

int CommMPI::process_rank() { return proc_rank; }
#endif
