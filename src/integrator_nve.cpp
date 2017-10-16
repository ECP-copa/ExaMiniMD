#include <integrator_nve.h>

IntegratorNVE::IntegratorNVE(System* s):Integrator(s) {
  dtv = system->dt;
  dtf = 0.5 * system->dt / system->mvv2e;
}

namespace {
  struct InitialIntegrateFunctor {
    t_x x;
    t_v v;
    t_f_const f;
    t_type_const type;
    t_mass_const mass;
    t_id id;
    int step;

    T_V_FLOAT dtf,dtv;
    InitialIntegrateFunctor(
      t_x& x_, t_v& v_, t_f& f_,
      t_type type_, t_mass mass_, t_id id_,
      T_V_FLOAT dtf_, T_V_FLOAT dtv_, int step_
    ):x(x_),v(v_),f(f_),type(type_),mass(mass_),
      id(id_),dtf(dtf_),dtv(dtv_),step(step_) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const T_INT& i) const {
      const T_V_FLOAT dtfm = dtf / mass(type(i));
      v(i,0) += dtfm * f(i,0);
      v(i,1) += dtfm * f(i,1);
      v(i,2) += dtfm * f(i,2);
      x(i,0) += dtv * v(i,0);
      x(i,1) += dtv * v(i,1);
      x(i,2) += dtv * v(i,2);
    }
  };
}

void IntegratorNVE::initial_integrate() {
  static int step =1;
  Kokkos::parallel_for("IntegratorNVE::initial_integrate",system->N_local,
                       InitialIntegrateFunctor(system->x, system->v, system->f,
                                               system->type, system->mass, system->id, dtf, dtv, step));
  step++;
}


namespace {
  struct FinalIntegrateFunctor {
    t_v v;
    t_f_const f;
    t_type_const type;
    t_mass_const mass;
    t_id id;
    int step;
    t_x x;

    T_V_FLOAT dtf,dtv;
    FinalIntegrateFunctor(
      t_v& v_, t_f& f_,
      t_type type_, t_mass mass_,
      T_V_FLOAT dtf_, T_V_FLOAT dtv_,
      t_id id_, int step_, t_x x_
    ):v(v_),f(f_),type(type_),mass(mass_),
      dtf(dtf_),dtv(dtv_),id(id_),step(step_),x(x_) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const T_INT& i) const {
      const T_V_FLOAT dtfm = dtf / mass(type(i));
      v(i,0) += dtfm * f(i,0);
      v(i,1) += dtfm * f(i,1);
      v(i,2) += dtfm * f(i,2);
    }
  };
}

void IntegratorNVE::final_integrate() {
  static int step = 1;
  Kokkos::parallel_for("IntegratorNVE::final_integrate",system->N_local, 
                       FinalIntegrateFunctor(system->v, system->f,
                                             system->type, system->mass, dtf, dtv, system->id, step, system->x));
  step++;
}

