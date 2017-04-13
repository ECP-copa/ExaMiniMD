#include <neighbor_csr.h>

#ifdef KOKKOS_ENABLE_CUDA
template struct NeighborCSR<Kokkos::CudaSpace>;
#endif
template struct NeighborCSR<Kokkos::HostSpace>;

