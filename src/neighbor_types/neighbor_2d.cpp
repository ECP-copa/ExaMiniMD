#include <neighbor_2d.h>

#ifdef KOKKOS_ENABLE_CUDA
template struct Neighbor2D<typename Kokkos::Cuda::memory_space>;
#endif
template struct Neighbor2D<Kokkos::HostSpace>;

