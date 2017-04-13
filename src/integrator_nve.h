#ifndef INTEGRATOR_NVE_H
#define INTEGRATOR_NVE_H

#include <types.h>
#include <integrator.h>

class IntegratorNVE: public Integrator {
  T_V_FLOAT dtv, dtf;

public:
  IntegratorNVE(System* s);
  void initial_integrate();
  void final_integrate();
};
#endif
