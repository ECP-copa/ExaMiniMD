#include <neighbor_csr_map_constr.h>

#ifdef KOKKOS_ENABLE_CUDA
template struct NeighborCSRMapConstr<Kokkos::CudaSpace>;
#endif
template struct NeighborCSRMapConstr<Kokkos::HostSpace>;

