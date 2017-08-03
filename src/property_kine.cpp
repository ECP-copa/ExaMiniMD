#include <property_kine.h>

KinE::KinE(Comm* comm_):comm(comm_) {}

T_V_FLOAT KinE::compute(System* system) {
  v = system->v;
  mass = system->mass;
  type = system->type;

  T_V_FLOAT KE; 
  Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::IndexType<T_INT>>(0,system->N_local), *this, KE);

  // Make sure I don't carry around references to data
  v = t_v();
  mass = t_mass();
  type = t_type();

  // Multiply by scaling factor (units based) to get to kinetic energy
  T_V_FLOAT factor = 0.5 * system->mvv2e;

  comm->reduce_float(&KE,1);
  return KE * factor;
}
