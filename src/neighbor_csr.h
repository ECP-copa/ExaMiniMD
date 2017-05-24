#ifdef MODULES_OPTION_CHECK
      if( (strcmp(argv[i+1], "CSR") == 0) )
        neighbor_type = NEIGH_CSR;
#endif
#ifdef MODULES_INSTANTIATION
    else if (input->neighbor_type == NEIGH_CSR) {
      neighbor = new NeighborCSR<t_neigh_mem_space>();
      neighbor->init(input->force_cutoff + input->neighbor_skin);
    }
#endif


#if !defined(MODULES_OPTION_CHECK) && !defined(MODULES_INSTANTIATION)
#include <neighbor.h>
#ifndef NEIGHBOR_CSR_H
#define NEIGHBOR_CSR_H
#include <Kokkos_StaticCrsGraph.hpp>
#include <system.h>
#include <binning.h>

template<class MemorySpace>
struct NeighListCSR : public Kokkos::StaticCrsGraph<T_INT,Kokkos::LayoutLeft,MemorySpace,T_INT> {
  struct NeighViewCSR {
    private:
      const T_INT* const ptr;
      const T_INT num_neighs;

    public:
      KOKKOS_INLINE_FUNCTION
      NeighViewCSR (const T_INT* const ptr_, const T_INT& num_neighs_):
        ptr(ptr_),num_neighs(num_neighs_) {}

      KOKKOS_INLINE_FUNCTION
      T_INT operator() (const T_INT& i) const { return ptr[i]; }

      KOKKOS_INLINE_FUNCTION
      T_INT get_num_neighs() const { return num_neighs; }
  };

  typedef NeighViewCSR t_neighs;

  NeighListCSR() :
    Kokkos::StaticCrsGraph<T_INT,Kokkos::LayoutLeft,MemorySpace,T_INT>() {}
  NeighListCSR (const NeighListCSR& rhs) :
    Kokkos::StaticCrsGraph<T_INT,Kokkos::LayoutLeft,MemorySpace,T_INT>(rhs) {}

  template<class EntriesType, class RowMapType>
  NeighListCSR (const EntriesType& entries_,const RowMapType& row_map_) :
    Kokkos::StaticCrsGraph<T_INT,Kokkos::LayoutLeft,MemorySpace,T_INT>( entries_, row_map_) {}


  KOKKOS_INLINE_FUNCTION
  T_INT get_num_neighs(const T_INT& i) const {
    return this->row_map(i+1) - this->row_map(i);
  }

  KOKKOS_INLINE_FUNCTION
  t_neighs get_neighs(const T_INT& i) const {
    const T_INT start = this->row_map(i);
    const T_INT end = this->row_map(i+1);
    return t_neighs(&this->entries(start),end-start);
  }
};

template<class MemorySpace>
class NeighborCSR: public Neighbor {

protected:
  T_X_FLOAT neigh_cut;
  t_x x;
  t_type type;
  t_id id;

  T_INT nbinx,nbiny,nbinz,nhalo;
  T_INT N_local;

  Kokkos::View<T_INT*, MemorySpace> num_neighs;
  Kokkos::View<T_INT*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Atomic> > num_neighs_atomic;
  Kokkos::View<T_INT*, MemorySpace> neigh_offsets;
  Kokkos::View<T_INT*, MemorySpace> neighs;
  bool skip_num_neigh_count, half_neigh;

  typename Binning::t_binoffsets bin_offsets;
  typename Binning::t_bincount bin_count;
  typename Binning::t_permute_vector permute_vector;



public:
  struct TagCreateOffsets {};
  struct TagCountNeighsFull {};
  struct TagFillNeighListFull {};
  struct TagCountNeighsHalf {};
  struct TagFillNeighListHalf {};

  typedef Kokkos::RangePolicy<TagCreateOffsets, Kokkos::IndexType<T_INT> > t_policy_co;
  typedef Kokkos::TeamPolicy<TagCountNeighsFull, Kokkos::IndexType<T_INT> , Kokkos::Schedule<Kokkos::Dynamic> > t_policy_cnf;
  typedef Kokkos::TeamPolicy<TagFillNeighListFull, Kokkos::IndexType<T_INT> , Kokkos::Schedule<Kokkos::Dynamic> > t_policy_fnlf;
  typedef Kokkos::TeamPolicy<TagCountNeighsHalf, Kokkos::IndexType<T_INT> , Kokkos::Schedule<Kokkos::Dynamic> > t_policy_cnh;
  typedef Kokkos::TeamPolicy<TagFillNeighListHalf, Kokkos::IndexType<T_INT> , Kokkos::Schedule<Kokkos::Dynamic> > t_policy_fnlh;

  typedef NeighListCSR<MemorySpace> t_neigh_list;

  t_neigh_list neigh_list;


  NeighborCSR():neigh_cut(0.0) {
    neigh_type = NEIGH_CSR;
  };
  ~NeighborCSR() {};

  void init(T_X_FLOAT neigh_cut_) { neigh_cut = neigh_cut_; };

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagCountNeighsFull&, const typename t_policy_cnf::member_type& team) const {
    const T_INT bx = team.league_rank()/(nbiny*nbinz) + nhalo;
    const T_INT by = (team.league_rank()/(nbinz)) % nbiny + nhalo;
    const T_INT bz = team.league_rank() % nbinz + nhalo;

    const T_INT i_offset = bin_offsets(bx,by,bz);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,bin_count(bx,by,bz)), [&] (const int bi) {
      const T_INT i = permute_vector(i_offset + bi);
      if(i>=N_local) return;
      const T_F_FLOAT x_i = x(i,0);
      const T_F_FLOAT y_i = x(i,1);
      const T_F_FLOAT z_i = x(i,2);
      const int type_i = type(i);

      int neigh_count = 0;
      for(int bx_j = bx-1; bx_j<bx+2; bx_j++)
      for(int by_j = by-1; by_j<by+2; by_j++)
      for(int bz_j = bz-1; bz_j<bz+2; bz_j++) {

        const T_INT j_offset = bin_offsets(bx_j,by_j,bz_j);
        int neigh_count_temp;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, bin_count(bx_j,by_j,bz_j)), [&] (const T_INT bj, int& count) {
          T_INT j = permute_vector(j_offset + bj);

          const T_F_FLOAT dx = x_i - x(j,0);
          const T_F_FLOAT dy = y_i - x(j,1);
          const T_F_FLOAT dz = z_i - x(j,2);
          const int type_j = type(j);
          const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

          if((rsq <= neigh_cut*neigh_cut) && (i!=j)) {
            count++;
          }
        },neigh_count_temp);
        neigh_count+=neigh_count_temp;
      }
      Kokkos::single(Kokkos::PerThread(team), [&] () {
        num_neighs(i) = neigh_count;
      });
    });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagFillNeighListFull&, const typename t_policy_fnlf::member_type& team) const {
    const T_INT bx = team.league_rank()/(nbiny*nbinz) + nhalo;
    const T_INT by = (team.league_rank()/(nbinz)) % nbiny + nhalo;
    const T_INT bz = team.league_rank() % nbinz + nhalo;

    const T_INT i_offset = bin_offsets(bx,by,bz);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,bin_count(bx,by,bz)), [&] (const int bi) {
      const T_INT i = permute_vector(i_offset + bi);
      if(i>=N_local) return;
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
            T_INT offset = Kokkos::atomic_fetch_add(&num_neighs(i),1) + neigh_offsets(i);
            neighs(offset) = j;
          }
        });
      }
    });
  }

  KOKKOS_INLINE_FUNCTION
   void operator() (const TagCountNeighsHalf&, const typename t_policy_cnh::member_type& team) const {
     const T_INT bx = team.league_rank()/(nbiny*nbinz) + nhalo;
     const T_INT by = (team.league_rank()/(nbinz)) % nbiny + nhalo;
     const T_INT bz = team.league_rank() % nbinz + nhalo;

     const T_INT i_offset = bin_offsets(bx,by,bz);
     Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,bin_count(bx,by,bz)), [&] (const int bi) {
       const T_INT i = permute_vector(i_offset + bi);
       if(i>=N_local) return;
       const T_F_FLOAT x_i = x(i,0);
       const T_F_FLOAT y_i = x(i,1);
       const T_F_FLOAT z_i = x(i,2);
       const int type_i = type(i);

       int neigh_count = 0;
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
           Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, bin_count(bx_j,by_j,bz_j)), [&] (const T_INT bj, int& count) {
             T_INT j = permute_vector(j_offset + bj);
             const T_F_FLOAT x_j = x(j,0);
             const T_F_FLOAT y_j = x(j,1);
             const T_F_FLOAT z_j = x(j,2);
             if( (j<N_local) && !((x_j > x_i)  || ((x_j == x_i) && ( (y_j>y_i) ||  ((y_j==y_i) && (z_j>z_i) )))))
               return;
             const T_F_FLOAT dx = x_i - x_j;
             const T_F_FLOAT dy = y_i - y_j;
             const T_F_FLOAT dz = z_i - z_j;

             const int type_j = type(j);
             const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

             if((rsq <= neigh_cut*neigh_cut)) {
               count++;
             }
           },neigh_count_temp);
         neigh_count+=neigh_count_temp;
       }
       Kokkos::single(Kokkos::PerThread(team), [&] () {
         num_neighs(i) = neigh_count;
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
       const T_F_FLOAT x_i = x(i,0);
       const T_F_FLOAT y_i = x(i,1);
       const T_F_FLOAT z_i = x(i,2);
       const int type_i = type(i);

       for(int bx_j = bx-1; bx_j<bx+2; bx_j++)
       for(int by_j = by-1; by_j<by+2; by_j++)
       for(int bz_j = bz-1; bz_j<bz+2; bz_j++) {
/*
         if( ( (bx_j<bx) || ((bx_j == bx) && ( (by_j>by) ||  ((by_j==by) && (bz_j>bz) )))) && 
             (bx_j>nhalo) && (bx_j<nbinx+nhalo-2) && 
             (by_j>nhalo) && (by_j<nbiny+nhalo-2) &&
             (bz_j>nhalo) && (bz_j<nbinz+nhalo-2) 
           ) continue;*/
         const T_INT j_offset = bin_offsets(bx_j,by_j,bz_j);

           Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, bin_count(bx_j,by_j,bz_j)), [&] (const T_INT bj) {
             T_INT j = permute_vector(j_offset + bj);
             const T_F_FLOAT x_j = x(j,0);
             const T_F_FLOAT y_j = x(j,1);
             const T_F_FLOAT z_j = x(j,2);
             if( (j<N_local) && !((x_j > x_i)  || ((x_j == x_i) && ( (y_j>y_i) ||  ((y_j==y_i) && (z_j>z_i) )))))
               return;
             const T_F_FLOAT dx = x_i - x_j;
             const T_F_FLOAT dy = y_i - y_j;
             const T_F_FLOAT dz = z_i - z_j;

             const int type_j = type(j);
             const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

             if((rsq <= neigh_cut*neigh_cut)) {
               T_INT offset = Kokkos::atomic_fetch_add(&num_neighs(i),1) + neigh_offsets(i);
               neighs(offset) = j;
             }
           });
       }
     });
   }
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagCreateOffsets&, const T_INT& i, T_INT& offset, const bool final) const {
    const T_INT count_i = num_neighs(i);
    if(final) {
      neigh_offsets(i) = offset;
      if(i==N_local-1)
        neigh_offsets(i+1) = offset+count_i;
    }
    offset += count_i;
  }

  void create_neigh_list(System* system, Binning* binning, bool half_neigh_, bool) {
    // Get some data handles
    N_local = system->N_local;
    x = system->x;
    type = system->type;
    id = system->id;
    half_neigh = half_neigh_;

    T_INT total_num_neighs;


    // Reset the neighbor count array
    if( num_neighs.extent(0) < N_local + 1 ) {
      num_neighs = Kokkos::View<T_INT*, MemorySpace>("NeighborsCSR::num_neighs", N_local + 1);
      neigh_offsets = Kokkos::View<T_INT*, MemorySpace>("NeighborsCSR::neigh_offsets", N_local + 1);
    } else
      Kokkos::deep_copy(num_neighs,0);
    num_neighs_atomic = num_neighs;


    // Create the pair list
    nhalo = binning->nhalo;
    nbinx = binning->nbinx - 2*nhalo;
    nbiny = binning->nbiny - 2*nhalo;
    nbinz = binning->nbinz - 2*nhalo;

    T_INT nbins = nbinx*nbiny*nbinz;

    bin_offsets = binning->binoffsets;
    bin_count = binning->bincount;
    permute_vector = binning->permute_vector;

    if(half_neigh)
      Kokkos::parallel_for("NeighborCSR::count_neighbors_half", t_policy_cnh(nbins,Kokkos::AUTO,8),*this);
    else
      Kokkos::parallel_for("NeighborCSR::count_neighbors_full", t_policy_cnf(nbins,Kokkos::AUTO,8),*this);
    Kokkos::fence();

    // Create the Offset list for neighbors of atoms
    Kokkos::parallel_scan("NeighborCSR::create_offsets", t_policy_co(0, N_local), *this);
    Kokkos::fence();

    // Get the total neighbor count
    Kokkos::View<T_INT,MemorySpace> d_total_num_neighs(neigh_offsets,N_local);
    Kokkos::deep_copy(total_num_neighs,d_total_num_neighs);

    // Resize NeighborList
    if( neighs.extent(0) < total_num_neighs )
      neighs = Kokkos::View<T_INT*, MemorySpace> ("NeighborCSR::neighs", total_num_neighs);

    // Copy entries from the PairList to the actual NeighborList
    Kokkos::deep_copy(num_neighs,0);

    if(half_neigh)
      Kokkos::parallel_for("NeighborCSR::fill_neigh_list_half",t_policy_fnlh(nbins,Kokkos::AUTO,8),*this);
    else
      Kokkos::parallel_for("NeighborCSR::fill_neigh_list_full",t_policy_fnlf(nbins,Kokkos::AUTO,8),*this);

    Kokkos::fence();

    // Create actual CSR NeighList
    neigh_list = t_neigh_list(
        Kokkos::View<T_INT*, MemorySpace>( neighs,     Kokkos::pair<T_INT,T_INT>(0,total_num_neighs)),
        Kokkos::View<T_INT*, MemorySpace>( neigh_offsets, Kokkos::pair<T_INT,T_INT>(0,N_local+1)));

  }

  t_neigh_list get_neigh_list() { return neigh_list; }
  const char* name() {return "NeighborCSR";}
};

#ifdef KOKKOS_ENABLE_CUDA
extern template struct NeighborCSR<Kokkos::CudaSpace>;
#endif
extern template struct NeighborCSR<Kokkos::HostSpace>;
#endif // #define NEIGHBOR_CSR_H
#endif // MODULES_OPTION_CHECK / MODULES_INSTANTIATION
