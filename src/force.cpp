#include<force.h>

Force::Force(char** args, System* system, bool half_neigh_):half_neigh(half_neigh_) {}

void Force::init_coeff(int nargs, char** args) {}
void Force::compute(System*, Binning*, Neighbor*) { }
const char* Force::name() { return "ForceNone"; }

