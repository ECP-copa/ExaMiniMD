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
const char* Comm::name() {return "InvalidComm";}

