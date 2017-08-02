#include <neighbor_csr.h>

#ifdef KOKKOS_ENABLE_CUDA
template struct NeighborCSR<typename Kokkos::Cuda::memory_space>;
#endif
template struct NeighborCSR<Kokkos::HostSpace>;

