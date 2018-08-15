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

#include<comm.h>
Comm::Comm(System* s, T_X_FLOAT comm_depth_):system(s),comm_depth(comm_depth_) {}
Comm::~Comm() {}
void Comm::init() {};
void Comm::exchange() {};
void Comm::exchange_halo() {};
void Comm::update_halo() {};
void Comm::update_force() {};
void Comm::reduce_float(T_FLOAT*, T_INT) {};
void Comm::reduce_int(T_INT*, T_INT) {};
void Comm::reduce_max_float(T_FLOAT*, T_INT) {};
void Comm::reduce_max_int(T_INT*, T_INT) {};
void Comm::reduce_min_float(T_FLOAT*, T_INT) {};
void Comm::reduce_min_int(T_INT*, T_INT) {};
void Comm::scan_int(T_INT*, T_INT) {};
void Comm::weighted_reduce_float(T_FLOAT* , T_INT* , T_INT ) {};
void Comm::create_domain_decomposition() {
  system->sub_domain_lo_x = 0.0;
  system->sub_domain_lo_y = 0.0;
  system->sub_domain_lo_z = 0.0;
  system->sub_domain_x = system->sub_domain_hi_x = system->domain_x;
  system->sub_domain_y = system->sub_domain_hi_y = system->domain_y;
  system->sub_domain_z = system->sub_domain_hi_z = system->domain_z;
};
int Comm::process_rank() {return 0;}
int Comm::num_processes() {return 1;}
void Comm::error(const char *errormsg) {
  printf("%s\n",errormsg);
  exit(1);
};
const char* Comm::name() {return "InvalidComm";}

