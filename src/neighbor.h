#ifndef NEIGHBOR_H
#define NEIGHBOR_H
#include <types.h>
#include <system.h>
#include <binning.h>

class Neighbor {
public:
  int neigh_type;
  Neighbor();
  virtual ~Neighbor();
  typedef Kokkos::View<int**> t_neigh_list;
  virtual void init(T_X_FLOAT neighcut);
  virtual void create_neigh_list(System* system, Binning* binning = NULL);
  t_neigh_list get_neigh_list();
  virtual const char* name();
};

#include <modules_neighbor.h>

#endif
