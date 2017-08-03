#include <types.h>

#include<system.h>
#include<integrator.h>
#include<force.h>
#include<neighbor.h>
#include<comm.h>
#include<input.h>
#include<binning.h>

class ExaMiniMD {
  public:
    System* system;
    Integrator* integrator;
    Force* force;
    Neighbor* neighbor;
    Comm* comm;
    Input* input;
    Binning* binning;

    ExaMiniMD();

    void init(int argc,char* argv[]);
       
    void run(int nsteps);

    void dump_binary(int);
    void check_correctness(int);

    void print_performance();

    void shutdown();
};

