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

#ifndef TYPES_H
#define TYPES_H
#include<Kokkos_Core.hpp>

// Module Types etc
// Units to be used
enum {UNITS_REAL,UNITS_LJ,UNITS_METAL};
// Lattice Type
enum {LATTICE_SC,LATTICE_FCC};
// Integrator Type
enum {INTEGRATOR_NVE};
// Binning Type
enum {BINNING_KKSORT};
// Comm Type
enum {COMM_SERIAL,COMM_MPI};
// Force Type
enum {FORCE_LJ, FORCE_LJ_IDIAL, FORCE_SNAP};
// Force Iteration Type
enum {FORCE_ITER_CELL_FULL, FORCE_ITER_NEIGH_FULL, FORCE_ITER_NEIGH_HALF};
// Neighbor Type
enum {NEIGH_NONE, NEIGH_CSR, NEIGH_CSR_MAPCONSTR, NEIGH_2D};
// Input File Type
enum {INPUT_LAMMPS};

// Macros to work around the fact that std::max/min is not available on GPUs
#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)

#define MAX_TYPES_STACKPARAMS 12

// Define Scalar Types
#ifndef T_INT
#define T_INT int
#endif

#ifndef T_FLOAT
#define T_FLOAT double
#endif
#ifndef T_X_FLOAT
#define T_X_FLOAT T_FLOAT
#endif
#ifndef T_V_FLOAT
#define T_V_FLOAT T_FLOAT
#endif
#ifndef T_F_FLOAT
#define T_F_FLOAT T_FLOAT
#endif

// Define Kokkos View Types
typedef Kokkos::View<T_X_FLOAT*[3],Kokkos::LayoutRight>       t_x;          // Positions
typedef Kokkos::View<const T_X_FLOAT*[3],Kokkos::LayoutRight> t_x_const;    // Positions
typedef Kokkos::View<const T_X_FLOAT*[3],Kokkos::LayoutRight,
    Kokkos::MemoryTraits<Kokkos::RandomAccess>> t_x_const_rnd;    // Positions
typedef Kokkos::View<T_V_FLOAT*[3]>       t_v;          // Velocities
typedef Kokkos::View<T_F_FLOAT*[3]>       t_f;          // Force
typedef Kokkos::View<T_F_FLOAT*[3],
#ifdef KOKKOS_ENABLE_SERIAL
    Kokkos::MemoryTraits<std::is_same<Kokkos::DefaultExecutionSpace,Kokkos::Serial>::value?0:Kokkos::Atomic>>
#else
    Kokkos::MemoryTraits<Kokkos::Atomic>>
#endif
        t_f_atomic;   // Force
typedef Kokkos::View<const T_F_FLOAT*[3]> t_f_const;    // Force

typedef Kokkos::View<int*>                t_type;       // Type (int is enough as type)
typedef Kokkos::View<const int*>          t_type_const; // Type (int is enough as type)
typedef Kokkos::View<const int*,
    Kokkos::MemoryTraits<Kokkos::RandomAccess>> t_type_const_rnd; // Type (int is enough as type)
typedef Kokkos::View<T_INT*>              t_id;         // ID
typedef Kokkos::View<const T_INT*>        t_id_const;   // ID
typedef Kokkos::View<T_FLOAT*>            t_q;          // Charge
typedef Kokkos::View<const T_FLOAT*>      t_q_const;    // Charge

typedef Kokkos::View<T_V_FLOAT*>          t_mass;       // Mass
typedef Kokkos::View<const T_V_FLOAT*>    t_mass_const; // Mass

typedef Kokkos::DefaultExecutionSpace::memory_space t_neigh_mem_space;

template<class Scalar>
struct t_scalar3 {
  Scalar x,y,z;

  KOKKOS_FORCEINLINE_FUNCTION
  t_scalar3() {
    x = 0; y = 0; z = 0;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  t_scalar3(const t_scalar3& rhs) {
    x = rhs.x; y = rhs.y; z = rhs.z;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  t_scalar3(const Scalar& x_, const Scalar& y_, const Scalar& z_ ) {
    x = x_; y = y_; z = z_;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  t_scalar3 operator= (const t_scalar3& rhs) {
    x = rhs.x; y = rhs.y; z = rhs.z;
    return *this;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  t_scalar3 operator= (const volatile t_scalar3& rhs) {
    x = rhs.x; y = rhs.y; z = rhs.z;
    return *this;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  t_scalar3 operator+= (const t_scalar3& rhs) {
    x += rhs.x; y += rhs.y; z += rhs.z;
    return *this;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  t_scalar3 operator+= (const volatile t_scalar3& rhs) volatile {
    x += rhs.x; y += rhs.y; z += rhs.z;
    return *this;
  }
};

template<class Scalar>
KOKKOS_FORCEINLINE_FUNCTION
t_scalar3<Scalar> operator +
  (const t_scalar3<Scalar>& a, const t_scalar3<Scalar>& b) {
  return t_scalar3<Scalar>(a.x+b.x,a.y+b.y,a.z+b.z);
}

template<class Scalar>
KOKKOS_FORCEINLINE_FUNCTION
t_scalar3<Scalar> operator *
  (const t_scalar3<Scalar>& a, const Scalar& b) {
  return t_scalar3<Scalar>(a.x*b,a.y*b,a.z*b);
}

template<class Scalar>
KOKKOS_FORCEINLINE_FUNCTION
t_scalar3<Scalar> operator *
  (const Scalar& b, const t_scalar3<Scalar>& a) {
  return t_scalar3<Scalar>(a.x*b,a.y*b,a.z*b);
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
#define EMD_ENABLE_GPU
#endif

#endif

