#include <neighbor.h>

Neighbor::Neighbor() {}

Neighbor::~Neighbor() {};

void Neighbor::init(T_X_FLOAT neighcut) {};

void Neighbor::create_neigh_list(System* system, Binning* binning) {};

Neighbor::t_neigh_list Neighbor::get_neigh_list() { return Neighbor::t_neigh_list(); };

