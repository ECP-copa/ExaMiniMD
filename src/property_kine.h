
#include <types.h>
#include <system.h>
#include <comm.h>

class KinE {
  private:
    t_v v;
    t_mass mass;
    t_type type;
    Comm* comm;
  public:
    KinE(Comm* comm_);

    T_V_FLOAT compute(System*);
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const T_INT& i, T_V_FLOAT& KE) const {
      KE += (v(i,0)*v(i,0) + v(i,1)*v(i,1) + v(i,2)*v(i,2)) * mass(type(i));
    }
};
