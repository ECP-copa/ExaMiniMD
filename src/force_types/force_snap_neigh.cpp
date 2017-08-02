#include<force_snap_neigh_impl.h>
#define FORCETYPE_DECLARE_TEMPLATE_MACRO(NeighType) ForceSNAP<NeighType>
#define FORCE_MODULES_TEMPLATE
#include<modules_neighbor.h>
#undef FORCE_MODULES_TEMPLATE
