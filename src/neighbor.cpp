#include <neighbor.h>

Neighbor::Neighbor():neigh_type(NEIGH_NONE) {}

Neighbor::~Neighbor() {};

void Neighbor::init(T_X_FLOAT neighcut) {};

void Neighbor::create_neigh_list(System* system, Binning* binning, bool half_neigh_ , bool ghost_neighs_) {};

Neighbor::t_neigh_list Neighbor::get_neigh_list() { return Neighbor::t_neigh_list(); };

const char* Neighbor::name() {return "NeighborNone";}

