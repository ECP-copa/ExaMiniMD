#include <examinimd.h>
#include <property_pote.h>

PotE::PotE(Comm* comm_):comm(comm_) {}

T_V_FLOAT PotE::compute(System* system, Binning* binning, Neighbor* neighbor, Force* force) {

  T_V_FLOAT PE; 
  force->compute(system,binning,neighbor,PE);

  comm->reduce_float(&PE,1);
  return PE;
}
