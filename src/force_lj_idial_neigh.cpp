#include<force_lj_idial_neigh_impl.h>
#define FORCETYPE_DECLARE_TEMPLATE_MACRO(NeighType) ForceLJIDialNeigh<NeighType>
#define FORCE_MODULES_TEMPLATE
#include<modules_neighbor.h>
#undef FORCE_MODULES_TEMPLATE
