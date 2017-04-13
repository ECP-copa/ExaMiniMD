#include <property_temperature.h>

Temperature::Temperature(Comm* comm_):comm(comm_) {}

T_V_FLOAT Temperature::compute(System* system) {
  v = system->v;
  mass = system->mass;
  type = system->type;

  T_V_FLOAT T; 
  Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::IndexType<T_INT>>(0,system->N_local), *this, T);

  // Make sure I don't carry around references to data
  v = t_v();
  mass = t_mass();
  type = t_type();

  // Multiply by scaling factor (units based) to get to temperature
  T_INT dof = 3 * system->N - 3;
  T_V_FLOAT factor = system->mvv2e / (1.0 * dof * system->boltz);

  comm->reduce_float(&T,1);
  return T * factor;
}
