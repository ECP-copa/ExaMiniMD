# ExaMiniMD

ExaMiniMD is a proxy applications and research vehicle for 
particle codes, in particular Molecular Dynamics (MD). Compared to 
previous MD proxy apps (MiniMD, COMD), its design is signicantly more 
modular in order to allow independent investigation of different aspects.
To achieve that the main components such as force calculation, 
communication, neighbor list concstruction and binning are derived 
classes whos main functionality is accessed via virtual functions. 
This allows developer to write a new derived class and drop it into the code
without touching much of the rest of the application.

These modules are included via a module header file. Those header files are
also used to inject the input parameter logic and instantiation logic into 
the main code. As an example look at modules_comm.h in conjunction with 
comm_serial.h and comm_mpi.h. 

In the future the plan is to provide focused miniApps with a subset of the 
available functionality for specific research purposes. 

This application is using Kokkos as a programming model which you can clone
from github via:
```
git clone https://github.com/kokkos/kokkos ~/kokkos
```

# Current Capabilities

### Force Fields:
 * Lennard Jones Cell List
 * Lennard Jones Neighbor List

### Neighbor List:
 * CSR NeighborList creation

### Integrator:
 * NVE (constant energy velocity verlet)

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
Currently ExaMiniMD requires the develop branch of Kokkos as of 
April 2017.
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

```
mpirun -np 2 -bind-to socket ./ExaMiniMD -il ../input/in.lj --comm-type MPI --kokkos-threads=12
```

