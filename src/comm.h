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

#if !defined(MODULES_OPTION_CHECK) && !defined(MODULES_INSTANTIATION)

#ifndef COMM_H
#define COMM_H
#include <types.h>
#include <system.h>
#include <binning.h>

class Comm {

protected:
  System* system;

  T_X_FLOAT comm_depth;

public:
  Comm(System* s, T_X_FLOAT comm_depth);
  virtual ~Comm();
  virtual void init();

  // Move particles which left local domain
  virtual void exchange();

  // Exchange ghost particles
  virtual void exchange_halo();

  // Update ghost particles
  virtual void update_halo();

  // Reverse communication of forces
  virtual void update_force();

  // Do a sum reduction over floats
  virtual void reduce_float(T_FLOAT* values, T_INT N);

  // Do a sum reduction over integers
  virtual void reduce_int(T_INT* values, T_INT N);

  // Do a max reduction over floats
  virtual void reduce_max_float(T_FLOAT* values, T_INT N);

  // Do a max reduction over integers
  virtual void reduce_max_int(T_INT* values, T_INT N);

  // Do a min reduction over floats
  virtual void reduce_min_float(T_FLOAT* values, T_INT N);

  // Do a min reduction over integers
  virtual void reduce_min_int(T_INT* values, T_INT N);

  // Do an inclusive scan over integers
  virtual void scan_int(T_INT* values, T_INT N);

  // Do a sum reduction over floats with weights
  virtual void weighted_reduce_float(T_FLOAT* values, T_INT* weight, T_INT N);

  // Create a processor grid
  virtual void create_domain_decomposition();

  // Get Processor rank
  virtual int process_rank();

  // Get number of processors
  virtual int num_processes();

  // Exit with error message
  virtual void error(const char *);

  // Get class name
  virtual const char* name();
};


#include<modules_comm.h>

#endif
#endif

