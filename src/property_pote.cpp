#include <examinimd.h>
#include <property_pote.h>

PotE::PotE(Comm* comm_):comm(comm_) {}

T_F_FLOAT PotE::compute(System* system, Binning* binning, Neighbor* neighbor, Force* force) {
  T_F_FLOAT PE; 
  PE = force->compute_energy(system,binning,neighbor);
  comm->reduce_float(&PE,1);
  return PE;
}
