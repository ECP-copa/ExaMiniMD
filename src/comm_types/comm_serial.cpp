//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimer in the documentation
//       and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//************************************************************************

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

void CommSerial::update_force() {
  //printf("Update Force\n");
  s=*system;
  ghost_offsets[0] = s.N_local;
  for(phase = 1; phase<6; phase++) {
    ghost_offsets[phase] = ghost_offsets[phase-1] + num_ghost[phase-1];
  }

  for(phase = 5; phase>=0; phase--) {
    pack_indicies = Kokkos::subview(pack_indicies_all,phase,Kokkos::ALL());

    Kokkos::parallel_for("CommSerial::halo_force_self",
      Kokkos::RangePolicy<TagHaloForceSelf, Kokkos::IndexType<T_INT> >(0,num_ghost[phase]),
      *this);
  }
};

const char* CommSerial::name() { return "CommSerial"; }
