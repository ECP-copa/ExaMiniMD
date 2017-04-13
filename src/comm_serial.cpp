#include<comm_serial.h>

CommSerial::CommSerial(System* s, T_X_FLOAT comm_depth):Comm(s,comm_depth) {
  printf("CommSerial\n");
  pack_count = Kokkos::View<int>("CommSerial::pack_count");
  pack_indicies_all = Kokkos::View<T_INT**,Kokkos::LayoutRight>("CommSerial::pack_indicies_all",6,0);
}

void CommSerial::exchange() {
  s = *system;
  N_local = system->N_local;

  Kokkos::parallel_for("CommSerial::exchange_self",
            Kokkos::RangePolicy<TagExchangeSelf, Kokkos::IndexType<T_INT> >(0,N_local), *this);
};

void CommSerial::exchange_halo() {

  N_local = system->N_local;
  N_ghost = 0;

  s = *system;

  for(phase = 0; phase < 6; phase ++) {
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());

    T_INT count = 0;
    Kokkos::deep_copy(pack_count,0);

    T_INT nparticles = N_local + N_ghost - ( (phase%2==1) ? num_ghost[phase-1]:0 );
    Kokkos::parallel_for("CommSerial::halo_exchange_self",
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
      Kokkos::resize(pack_indicies_all,6,count*1.1);
      pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());
      redo = true;
    }
    if(redo) {
      Kokkos::deep_copy(pack_count,0);
      Kokkos::parallel_for("CommSerial::halo_exchange_self",
                Kokkos::RangePolicy<TagHaloSelf, Kokkos::IndexType<T_INT> >(0,nparticles),
                *this);
    }

    num_ghost[phase] = count;

    N_ghost += count;
  }

  system->N_ghost = N_ghost;
};

void CommSerial::update_halo() {
  N_ghost = 0;
  s=*system;
  for(phase = 0; phase<6; phase++) {
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());

    Kokkos::parallel_for("CommSerial::halo_update_self",
      Kokkos::RangePolicy<TagHaloUpdateSelf, Kokkos::IndexType<T_INT> >(0,num_ghost[phase]),
      *this);
    N_ghost += num_ghost[phase];
  }
};

const char* CommSerial::name() { return "CommSerial"; }
