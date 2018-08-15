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

#include <types.h>
#include <system.h>
#include <comm.h>

class ItemizedFile {
public:
  char*** words;
  int max_nlines;
  int nlines;
  int words_per_line;
  int max_word_size;
  ItemizedFile();
  void allocate_words(int num_lines);
  void free_words();
  void print_line(int line);
  int words_in_line(int line);
  void print();
  void add_line(const char* const line);
};

// Class replicating LAMMPS Random velocity initialization with GEOM option
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836

class LAMMPS_RandomVelocityGeom {
 private:
  int seed;
 public:

  KOKKOS_INLINE_FUNCTION
  LAMMPS_RandomVelocityGeom (): seed(0) {};

  KOKKOS_INLINE_FUNCTION
  double uniform() {
    int k = seed/IQ;
    seed = IA*(seed-k*IQ) - IR*k;
    if (seed < 0) seed += IM;
    double ans = AM*seed;
    return ans;
  }

  KOKKOS_INLINE_FUNCTION
  double gaussian() {
    double v1,v2,rsq;
    do {
      v1 = 2.0*uniform()-1.0;
      v2 = 2.0*uniform()-1.0;
      rsq = v1*v1 + v2*v2;
    } while ((rsq >= 1.0) || (rsq == 0.0));

    const double fac = sqrt(-2.0*log(rsq)/rsq);
    return v2*fac;
  }

  KOKKOS_INLINE_FUNCTION
  void reset(int ibase, double *coord)
  {
    int i;

    char *str = (char *) &ibase;
    int n = sizeof(int);

    unsigned int hash = 0;
    for (i = 0; i < n; i++) {
      hash += str[i];
      hash += (hash << 10);
      hash ^= (hash >> 6);
    }

    str = (char *) coord;
    n = 3 * sizeof(double);
    for (i = 0; i < n; i++) {
      hash += str[i];
      hash += (hash << 10);
      hash ^= (hash >> 6);
    }

    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    // keep 31 bits of unsigned int as new seed
    // do not allow seed = 0, since will cause hang in gaussian()

    seed = hash & 0x7ffffff;
    if (!seed) seed = 1;

    // warm up the RNG

    for (i = 0; i < 5; i++) uniform();
  }
};


class Input {
 private:
  bool timestepflag; // input timestep?  
 public:
  System* system;
  
  char* input_file;
  int input_file_type;
  ItemizedFile input_data;

  int units;

  int lattice_style;
  double lattice_constant, lattice_offset_x, lattice_offset_y, lattice_offset_z;
  int lattice_nx, lattice_ny, lattice_nz;

  double temperature_target;
  int temperature_seed;

  int integrator_type;
  int nsteps;

  int binning_type;

  int comm_type;
  int comm_exchange_rate;
  int comm_newton;

  int force_type;
  int force_iteration_type;
  int force_line;
  T_F_FLOAT force_cutoff;
  Kokkos::View<int*,Kokkos::HostSpace> force_coeff_lines;


  T_F_FLOAT neighbor_skin; 
  int neighbor_type;
  
  int thermo_rate, dumpbinary_rate, correctness_rate;
  bool dumpbinaryflag, correctnessflag;
  char *dumpbinary_path, *reference_path, *correctness_file;
 
public:
  Input(System* s);
  void read_command_line_args(int argc, char* argv[]);
  void read_file(const char* filename = NULL);
  void read_lammps_file(const char* filename);
  void check_lammps_command(int line);
  void create_lattice(Comm* comm);
};
