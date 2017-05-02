#ifndef FORCE_H
#define FORCE_H
#include<types.h>
#include<system.h>
#include<binning.h>
#include<neighbor.h>

class Force {
public:
  bool half_neigh;
  Force(char** args, System* system, bool half_neigh_);

  virtual void init_coeff(int nargs, char** args);

  virtual void compute(System* system, Binning* binning, Neighbor* neigh);
  virtual T_F_FLOAT compute_energy(System* system, Binning* binning, Neighbor* neigh){return 0.0;}; // Only needed for thermo output

  virtual const char* name();
};

#include<modules_force.h>
#endif
