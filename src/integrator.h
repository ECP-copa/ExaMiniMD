#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#include <types.h>
#include <system.h>

class Integrator {
public:
  System* system;

  Integrator(System* s);
  virtual ~Integrator();
  T_V_FLOAT timestep_size;

  virtual void initial_integrate();
  virtual void final_integrate();
};

#include<modules_integrator.h>
#endif
