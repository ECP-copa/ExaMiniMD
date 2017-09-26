// Runtime Check for this Neighbor Class
#ifdef MODULES_OPTION_CHECK
      if( (strcmp(argv[i+1], "2D") == 0) )
        neighbor_type = NEIGH_2D;
#endif

// Instantiation and Init of this class
#ifdef NEIGHBOR_MODULES_INSTANTIATION
    else if (input->neighbor_type == NEIGH_2D) {
      neighbor = new Neighbor2D<t_neigh_mem_space>();
      neighbor->init(input->force_cutoff + input->neighbor_skin);
    }
#endif

// Add Force Instantiation case
#if defined(FORCE_MODULES_INSTANTIATION)
      case NEIGH_2D: force = new FORCETYPE_ALLOCATION_MACRO(Neighbor2D<t_neigh_mem_space>); break;
#endif

// Add Force declaration line
#if defined(FORCE_MODULES_EXTERNAL_TEMPLATE)
      extern template class FORCETYPE_DECLARE_TEMPLATE_MACRO(Neighbor2D<t_neigh_mem_space>);
#endif

// Add Force Template Instantiation line
#if defined(FORCE_MODULES_TEMPLATE)
      template class FORCETYPE_DECLARE_TEMPLATE_MACRO(Neighbor2D<t_neigh_mem_space>);
#endif

// Making sure we are not just instantiating some Option
#if !defined(MODULES_OPTION_CHECK) && \
    !defined(NEIGHBOR_MODULES_INSTANTIATION) && \
    !defined(FORCE_MODULES_INSTANTIATION) && \
    !defined(FORCE_MODULES_EXTERNAL_TEMPLATE) && \
    !defined(FORCE_MODULES_TEMPLATE)
#include <neighbor.h>
#ifndef NEIGHBOR_2D_H
#define NEIGHBOR_2D_H
#include <system.h>
#include <binning.h>

template<class MemorySpace>
struct NeighList2D {
  struct NeighView2D {
    private:
      T_INT* const ptr;
      const T_INT num_neighs;
      const T_INT stride;

    public:
      KOKKOS_INLINE_FUNCTION
      NeighView2D (T_INT* const ptr_, const T_INT& num_neighs_, const T_INT& stride_):
        ptr(ptr_),num_neighs(num_neighs_),stride(stride_) {}

      KOKKOS_INLINE_FUNCTION
      T_INT& operator() (const T_INT& i) const { return ptr[i*stride]; }

      KOKKOS_INLINE_FUNCTION
      T_INT get_num_neighs() const { return num_neighs; }
  };

  typedef NeighView2D t_neighs;

  NeighList2D() {
    maxneighs = 16;
  }

  KOKKOS_INLINE_FUNCTION
  T_INT get_num_neighs(const T_INT& i) const {
    return this->num_neighs(i);
  }

  KOKKOS_INLINE_FUNCTION
  t_neighs get_neighs(const T_INT& i) const {
    return t_neighs(&this->neighs(i,0),this->num_neighs(i),&this->neighs(i,1)-&this->neighs(i,0));
  }

  T_INT maxneighs;

  Kokkos::View<T_INT*, MemorySpace> num_neighs;
  Kokkos::View<T_INT**, MemorySpace> neighs;
};

template<class MemorySpace>
class Neighbor2D: public Neighbor {

protected:
  T_X_FLOAT neigh_cut;
  t_x x;
  t_type type;
  t_id id;

  T_INT nbinx,nbiny,nbinz,nhalo;
  T_INT N_local;

  bool skip_num_neigh_count, half_neigh;

  typename Binning::t_binoffsets bin_offsets;
  typename Binning::t_bincount bin_count;
  typename Binning::t_permute_vector permute_vector;

  Kokkos::View<T_INT, MemorySpace> resize,new_maxneighs;
  typename Kokkos::View<T_INT, MemorySpace>::HostMirror h_resize,h_new_maxneighs;


public:
  struct TagFillNeighListFull {};
  struct TagFillNeighListHalf {};

  typedef Kokkos::TeamPolicy<TagFillNeighListFull, Kokkos::IndexType<T_INT> , Kokkos::Schedule<Kokkos::Dynamic> > t_policy_fnlf;
  typedef Kokkos::TeamPolicy<TagFillNeighListHalf, Kokkos::IndexType<T_INT> , Kokkos::Schedule<Kokkos::Dynamic> > t_policy_fnlh;

  typedef NeighList2D<MemorySpace> t_neigh_list;

  t_neigh_list neigh_list;


  Neighbor2D():neigh_cut(0.0) {
    neigh_type = NEIGH_2D;

    resize = Kokkos::View<T_INT, MemorySpace> ("Neighbor2D::resize");
    new_maxneighs = Kokkos::View<T_INT, MemorySpace> ("Neighbor2D::new_maxneighs");

    h_resize = Kokkos::create_mirror_view(resize);
    h_new_maxneighs = Kokkos::create_mirror_view(new_maxneighs);
  };
  ~Neighbor2D() {};

  void init(T_X_FLOAT neigh_cut_) {
    neigh_cut = neigh_cut_;

    // Create actual 2D NeighList
    neigh_list = t_neigh_list();
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagFillNeighListFull&, const typename t_policy_fnlf::member_type& team) const {
    const T_INT bx = team.league_rank()/(nbiny*nbinz) + nhalo;
    const T_INT by = (team.league_rank()/(nbinz)) % nbiny + nhalo;
    const T_INT bz = team.league_rank() % nbinz + nhalo;

    const T_INT i_offset = bin_offsets(bx,by,bz);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,bin_count(bx,by,bz)), [&] (const int bi) {
      const T_INT i = permute_vector(i_offset + bi);
      if(i>=N_local) return;
      const typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);
      const T_F_FLOAT x_i = x(i,0);
      const T_F_FLOAT y_i = x(i,1);
      const T_F_FLOAT z_i = x(i,2);
      const int type_i = type(i);

      for(int bx_j = bx-1; bx_j<bx+2; bx_j++)
      for(int by_j = by-1; by_j<by+2; by_j++)
      for(int bz_j = bz-1; bz_j<bz+2; bz_j++) {

        const T_INT j_offset = bin_offsets(bx_j,by_j,bz_j);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, bin_count(bx_j,by_j,bz_j)), [&] (const T_INT bj) {
          T_INT j = permute_vector(j_offset + bj);

          const T_F_FLOAT dx = x_i - x(j,0);
          const T_F_FLOAT dy = y_i - x(j,1);
          const T_F_FLOAT dz = z_i - x(j,2);
          const int type_j = type(j);
          const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

          if((rsq <= neigh_cut*neigh_cut) && (i!=j)) {
            T_INT n = Kokkos::atomic_fetch_add(&neigh_list.num_neighs(i),1); // n is non-incremented (old) value, could use parallel_scan instead
            if (n < neigh_list.maxneighs)
              neighs_i(n) = j;
          }
        });
      }
       Kokkos::single(Kokkos::PerThread(team), [&] () {
         const T_INT num_neighs_i = neigh_list.num_neighs(i);
         if (num_neighs_i > neigh_list.maxneighs) {
           resize() = 1;
           new_maxneighs() = num_neighs_i; // may resize more times than necessary, but probably less expensive than using an atomic max
         }
       });
    });
  }

  KOKKOS_INLINE_FUNCTION
   void operator() (const TagFillNeighListHalf&, const typename t_policy_fnlh::member_type& team) const {
     const T_INT bx = team.league_rank()/(nbiny*nbinz) + nhalo;
     const T_INT by = (team.league_rank()/(nbinz)) % nbiny + nhalo;
     const T_INT bz = team.league_rank() % nbinz + nhalo;


     const T_INT i_offset = bin_offsets(bx,by,bz);
     Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,bin_count(bx,by,bz)), [&] (const int bi) {
       const T_INT i = permute_vector(i_offset + bi);
       if(i>=N_local) return;
       const typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);
       const T_F_FLOAT x_i = x(i,0);
       const T_F_FLOAT y_i = x(i,1);
       const T_F_FLOAT z_i = x(i,2);
       const int type_i = type(i);

       for(int bx_j = bx-1; bx_j<bx+2; bx_j++)
       for(int by_j = by-1; by_j<by+2; by_j++)
       for(int bz_j = bz-1; bz_j<bz+2; bz_j++) {

       /*  if( ( (bx_j<bx) || ((bx_j == bx) && ( (by_j>by) ||  ((by_j==by) && (bz_j>bz) )))) &&    
             (bx_j>=nhalo) && (bx_j<nbinx+nhalo-1) &&    
             (by_j>=nhalo) && (by_j<nbiny+nhalo-1) &&
             (bz_j>=nhalo) && (bz_j<nbinz+nhalo-1)    
           ) continue;
*/
         const T_INT j_offset = bin_offsets(bx_j,by_j,bz_j);
         int neigh_count_temp;
           Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, bin_count(bx_j,by_j,bz_j)), [&] (const T_INT bj) {
             T_INT j = permute_vector(j_offset + bj);
             const T_F_FLOAT x_j = x(j,0);
             const T_F_FLOAT y_j = x(j,1);
             const T_F_FLOAT z_j = x(j,2);
             if( ((j==i) || (j<N_local || comm_newton==true)) && !((x_j > x_i)  || ((x_j == x_i) && ( (y_j>y_i) ||  ((y_j==y_i) && (z_j>z_i) )))))
               return;
             const T_F_FLOAT dx = x_i - x_j;
             const T_F_FLOAT dy = y_i - y_j;
             const T_F_FLOAT dz = z_i - z_j;

             const int type_j = type(j);
             const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

             if((rsq <= neigh_cut*neigh_cut)) {
               T_INT n = Kokkos::atomic_fetch_add(&neigh_list.num_neighs(i),1); // n is non-incremented (old) value, could use parallel_scan instead
               if (n < neigh_list.maxneighs)
                 neighs_i(n) = j;
             }
           });
       }
       Kokkos::single(Kokkos::PerThread(team), [&] () {
         const T_INT num_neighs_i = neigh_list.num_neighs(i);
         if (num_neighs_i > neigh_list.maxneighs) {
           resize() = 1;
           new_maxneighs() = num_neighs_i; // may resize more times than necessary, but probably less expensive than using an atomic max
         }
       });
     });
   }

  void create_neigh_list(System* system, Binning* binning, bool half_neigh_, bool) {
    // Get some data handles
    N_local = system->N_local;
    x = system->x;
    type = system->type;
    id = system->id;
    half_neigh = half_neigh_;

    // Reset the neighbor count array
    if( neigh_list.num_neighs.extent(0) < N_local + 1 )
      neigh_list.num_neighs = Kokkos::View<T_INT*, MemorySpace>("Neighbors2D::num_neighs", N_local + 1);

    // Create the pair list
    nhalo = binning->nhalo;
    nbinx = binning->nbinx - 2*nhalo;
    nbiny = binning->nbiny - 2*nhalo;
    nbinz = binning->nbinz - 2*nhalo;

    T_INT nbins = nbinx*nbiny*nbinz;

    bin_offsets = binning->binoffsets;
    bin_count = binning->bincount;
    permute_vector = binning->permute_vector;

    do {

      // Resize NeighborList
      if( neigh_list.neighs.extent(0) < N_local + 1 || neigh_list.neighs.extent(1) < neigh_list.maxneighs )
        neigh_list.neighs = Kokkos::View<T_INT**, MemorySpace> ("Neighbor2D::neighs", N_local + 1, neigh_list.maxneighs);


      // Fill the NeighborList
      Kokkos::deep_copy(neigh_list.num_neighs,0);
      Kokkos::deep_copy(resize,0);

      if(half_neigh)
        Kokkos::parallel_for("Neighbor2D::fill_neigh_list_half",t_policy_fnlh(nbins,Kokkos::AUTO,8),*this);
      else
        Kokkos::parallel_for("Neighbor2D::fill_neigh_list_full",t_policy_fnlf(nbins,Kokkos::AUTO,8),*this);


      Kokkos::fence();

      Kokkos::deep_copy(h_resize,resize);

      if (h_resize()) {
        Kokkos::deep_copy(h_new_maxneighs, new_maxneighs);
        neigh_list.maxneighs = h_new_maxneighs() * 1.2;
      }
    }
    while (h_resize());
  }

  t_neigh_list get_neigh_list() { return neigh_list; }
  const char* name() {return "Neighbor2D";}
};

template<>
struct NeighborAdaptor<NEIGH_2D> {
  typedef Neighbor2D<t_neigh_mem_space> type;
};

extern template struct Neighbor2D<t_neigh_mem_space>;
#endif // #define NEIGHBOR_2D_H
#endif // MODULES_OPTION_CHECK / MODULES_INSTANTIATION
