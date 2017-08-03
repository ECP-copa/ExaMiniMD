#include <types.h>
#include <system.h>
#include <comm.h>

class PotE {
  private:
    Comm* comm;
  public:
    PotE(Comm* comm_);

    T_F_FLOAT compute(System*, Binning*, Neighbor*, Force*);
};
