#include<binning_kksort.h>

BinningKKSort::BinningKKSort(System* s): Binning(s) {}

namespace {
  //This needs to be multi dimensional Range Policy later
  template<class BinCount1D, class BinOffsets1D, class BinCount3D, class BinOffsets3D>
  struct BinningKKSort_AssignOffsets {
    BinCount1D bin_count_1d;
    BinOffsets1D bin_offsets_1d;
    BinCount3D bin_count_3d;
    BinOffsets3D bin_offsets_3d;

    T_INT nbinx,nbiny,nbinz;
    BinningKKSort_AssignOffsets(BinCount1D bc1d, BinOffsets1D bo1d, 
                                BinCount3D bc3d, BinOffsets3D bo3d,
                                T_INT nx, T_INT ny, T_INT nz):
            bin_count_1d(bc1d),bin_offsets_1d(bo1d),bin_count_3d(bc3d),bin_offsets_3d(bo3d),
            nbinx(nx),nbiny(ny),nbinz(nz) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const T_INT& i) const {
      T_INT ix = i/(nbiny*nbinz);
      T_INT iy = (i/nbinz)%nbiny;
      T_INT iz = i%nbinz;

      bin_offsets_3d(ix,iy,iz) = bin_offsets_1d(i);
      bin_count_3d(ix,iy,iz) = bin_count_1d(i);
    }
  };
}

void BinningKKSort::create_binning(T_X_FLOAT dx_in, T_X_FLOAT dy_in, T_X_FLOAT dz_in, int halo_depth, bool do_local, bool do_ghost, bool sort) {
  if(do_local||do_ghost) {
    nhalo = halo_depth;
    std::pair<T_INT,T_INT> range(do_local?0:system->N_local,
                                 do_ghost?system->N_local+system->N_ghost:system->N_local);

    nbinx = T_INT(system->sub_domain_x/dx_in);
    nbiny = T_INT(system->sub_domain_y/dy_in);
    nbinz = T_INT(system->sub_domain_z/dz_in);

    if(nbinx == 0) nbinx = 1;
    if(nbiny == 0) nbiny = 1;
    if(nbinz == 0) nbinz = 1;

    T_X_FLOAT dx = system->sub_domain_x/nbinx;
    T_X_FLOAT dy = system->sub_domain_y/nbiny;
    T_X_FLOAT dz = system->sub_domain_z/nbinz;

    nbinx += 2*halo_depth;
    nbiny += 2*halo_depth;
    nbinz += 2*halo_depth;

    T_X_FLOAT eps = dx/1000;
    minx = -dx * halo_depth - eps + system->sub_domain_lo_x;
    maxx =  dx * halo_depth + eps + system->sub_domain_hi_x;
    miny = -dy * halo_depth - eps + system->sub_domain_lo_y;
    maxy =  dy * halo_depth + eps + system->sub_domain_hi_y;
    minz = -dz * halo_depth - eps + system->sub_domain_lo_z;
    maxz =  dz * halo_depth + eps + system->sub_domain_hi_z;

    T_INT nbin[3] = {nbinx,nbiny,nbinz};
    T_X_FLOAT min[3] = {minx,miny,minz};
    T_X_FLOAT max[3] = {maxx,maxy,maxz};

    t_binop binop(nbin,min,max);

    auto x = Kokkos::subview(system->x,range,Kokkos::ALL);

    sorter = t_sorter(x,binop);

    sorter.create_permute_vector();

    permute_vector = sorter.get_permute_vector();

    typename t_sorter::bin_count_type bin_count_1d = sorter.get_bin_count();
    typename t_sorter::offset_type bin_offsets_1d = sorter.get_bin_offsets();

    bincount = t_bincount("Binning::bincount",nbinx,nbiny,nbinz);
    binoffsets = t_binoffsets("Binning::binoffsets",nbinx,nbiny,nbinz);

    Kokkos::parallel_for("Binning::AssignOffsets",nbinx*nbiny*nbinz,
                    BinningKKSort_AssignOffsets<t_sorter::bin_count_type,t_sorter::offset_type,
                                                t_bincount, t_binoffsets>(bin_count_1d,bin_offsets_1d,
                                                                          bincount,binoffsets,
                                                                          nbinx,nbiny,nbinz));
    if(sort) {
      sorter.sort(x);
      auto v = Kokkos::subview(system->v,range,Kokkos::ALL);
      sorter.sort(v);
      auto f = Kokkos::subview(system->f,range,Kokkos::ALL);
      sorter.sort(f);
      t_type type(system->type,range);
      sorter.sort(type);
      t_id id(system->id,range);
      sorter.sort(id);
      t_q q(system->q,range);
      sorter.sort(q);
    }
  }
}

