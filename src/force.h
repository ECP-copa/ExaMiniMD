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
  virtual void compute(System* system, Binning* binning, Neighbor* neigh, T_V_FLOAT& PE){}; // base-class definition should be removed eventually

  virtual const char* name();
};

#include<modules_force.h>
#endif
