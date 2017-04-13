#ifndef TYPES_H
#define TYPES_H
#include<Kokkos_Core.hpp>

// Module Types etc
// Units to be used
enum {UNITS_REAL,UNITS_LJ};
// Lattice Type
enum {LATTICE_SC,LATTICE_FCC};
// Integrator Type
enum {INTEGRATOR_NVE};
// Binning Type
enum {BINNING_KKSORT};
// Comm Type
enum {COMM_SERIAL,COMM_MPI};
// Force Type
enum {FORCE_LJ_CELL,FORCE_LJ_NEIGH};
// Force Iteration Type
enum {FORCE_ITER_CELL_FULL, FORCE_ITER_NEIGH_FULL};
// Neighbor Type
enum {NEIGH_NONE, NEIGH_CSR_FULL};
// Input File Type
enum {INPUT_LAMMPS};


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
typedef Kokkos::View<T_X_FLOAT*[3]>       t_x;          // Positions
typedef Kokkos::View<const T_X_FLOAT*[3]> t_x_const;    // Positions
typedef Kokkos::View<T_V_FLOAT*[3]>       t_v;          // Velocities
typedef Kokkos::View<T_F_FLOAT*[3]>       t_f;          // Force
typedef Kokkos::View<const T_F_FLOAT*[3]> t_f_const;    // Force

typedef Kokkos::View<int*>                t_type;       // Type (int is enough as type)
typedef Kokkos::View<const int*>          t_type_const; // Type (int is enough as type)
typedef Kokkos::View<T_INT*>              t_id;         // ID
typedef Kokkos::View<const T_INT*>        t_id_const;   // ID
typedef Kokkos::View<T_FLOAT*>            t_q;          // Charge
typedef Kokkos::View<const T_FLOAT*>      t_q_const;    // Charge

typedef Kokkos::View<T_V_FLOAT*>          t_mass;       // Mass
typedef Kokkos::View<const T_V_FLOAT*>    t_mass_const; // Mass

typedef Kokkos::DefaultExecutionSpace::memory_space t_neigh_mem_space;
#endif

