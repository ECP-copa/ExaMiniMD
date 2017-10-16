# ExaMiniMD

ExaMiniMD is a proxy application and research vehicle for 
particle codes, in particular Molecular Dynamics (MD). Compared to 
previous MD proxy apps (MiniMD, COMD), its design is significantly more 
modular in order to allow independent investigation of different aspects.
To achieve that the main components such as force calculation, 
communication, neighbor list construction and binning are derived 
classes whose main functionality is accessed via virtual functions. 
This allows a developer to write a new derived class and drop it into the code
without touching much of the rest of the application.

These modules are included via a module header file. Those header files are
also used to inject the input parameter logic and instantiation logic into 
the main code. As an example, look at modules_comm.h in conjunction with 
comm_serial.h and comm_mpi.h. 

In the future the plan is to provide focused miniApps with a subset of the 
available functionality for specific research purposes. 

This implementation uses the Kokkos programming model, which you can clone
from github via:
```
git clone https://github.com/kokkos/kokkos ~/kokkos
```

# Current Capabilities

### Force Fields:
 * Lennard-Jones Cell List
 * Lennard-Jones Neighbor List

### Neighbor List:
 * 2D NeighborList creation
 * CSR NeighborList creation

### Integrator:
 * NVE (constant energy velocity-Verlet)

### Communication
 * Serial
 * MPI

### Binning:
 * Kokkos Sort Binning

### Input:
 * Restricted LAMMPS input files

# Compilation

ExaMiniMD utilizes the standard GNU Make build system of Kokkos. For
detailed information about the Kokkos build process please refer to 
documentation of Kokkos at github.com/kokkos/kokkos
ExaMiniMD requires Kokkos version 2.03 (April 2017) as a minimum.
Here are some quickstart information which assume that Kokkos was 
cloned into ${HOME}/kokkos (see above) and you are in the "src"
directory:

Intel Sandy-Bridge CPU / Serial / MPI:
```
  make -j KOKKOS_ARCH=SNB KOKKOS_DEVICES=Serial CXX=mpicxx MPI=1
```

Intel Haswell CPU / Pthread / No MPI:
```
  make -j KOKKOS_ARCH=HSW KOKKOS_DEVICES=Pthread CXX=clang MPI=0
```

IBM Power8 CPU / OpenMP / MPI
```
  make -j KOKKOS_ARCH=Power8 KOKKOS_DEVICES=OpenMP CXX=mpicxx
```

IBM Power8 CPU + NVIDIA P100 / CUDA / MPI (OpenMPI)
```
  export OMPI_CXX=[KOKKOS_PATH]/bin/nvcc_wrapper
  make -j KOKKOS_ARCH=Power8,Pascal60 KOKKOS_DEVICES=Cuda CXX=mpicxx
```

# Running

Currently ExaMiniMD can only get input from LAMMPS input files with a 
restricted set of LAMMPS commands. An example input file is provided in the
input directory. Assuming you build in the src directory run:

To run 2 MPI tasks, with 12 threads per task:
```
mpirun -np 2 -bind-to socket ./ExaMiniMD -il ../input/in.lj --comm-type MPI --kokkos-threads=12
```

To run in serial, writing binary output every timestep to ReferenceDir
```
./ExaMiniMD -il ../input/in.lj --kokkos-threads=1 --binarydump 1 ReferenceDir 
```

To run in serial with 2 threads, checking correctness every timestep against ReferenceDir
```
./ExaMiniMD -il ../input/in.lj --kokkos-threads=2 --correctness 1 ReferenceDir correctness.dat 
```

