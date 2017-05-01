#include <types.h>
#include <system.h>
#include <comm.h>

class PotE {
  private:
    t_v v;
    t_mass mass;
    t_type type;
    Comm* comm;
  public:
    PotE(Comm* comm_);

    T_V_FLOAT compute(System*, Binning*, Neighbor*, Force*);
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const T_INT& i, T_V_FLOAT& PE) const {
      PE += (v(i,0)*v(i,0) + v(i,1)*v(i,1) + v(i,2)*v(i,2)) * mass(type(i));
    }
};
