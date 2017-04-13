#include<integrator.h>
Integrator::Integrator(System* p):system(p) {}
Integrator::~Integrator() {}

void Integrator::initial_integrate() {}
void Integrator::final_integrate() {}
