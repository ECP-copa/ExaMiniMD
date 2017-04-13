#ifndef FORCE_H
#define FORCE_H
#include<types.h>
#include<system.h>
#include<binning.h>
#include<neighbor.h>

class Force {
public:
  Force(char** args, System* system);

  virtual void init_coeff(int nargs, char** args);

  virtual void compute(System* system, Binning* binning, Neighbor* neigh);
};

#include<modules_force.h>
#endif
