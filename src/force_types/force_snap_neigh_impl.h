//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimer in the documentation
//       and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//************************************************************************

/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "force_snap_neigh.h"
// Headers added by AJP to fix compile issues
//#include "pair_snap_kokkos.h"
//#include "neighbor_kokkos.h"


#define MAXLINE 1024
#define MAXWORD 3

// Outstanding issues with quadratic term
// 1. there seems to a problem with compute_optimized energy calc
// it does not match compute_regular, even when quadratic coeffs = 0

//static double t1 = 0.0;
//static double t2 = 0.0;
//static double t3 = 0.0;
//static double t4 = 0.0;
//static double t5 = 0.0;
//static double t6 = 0.0;
//static double t7 = 0.0;
/* ---------------------------------------------------------------------- */

// AJP, Stan, For SNAP update:

//int num_teams;
//int vector_length_;
//static constexpr int vector_length = vector_length_;
//template <class Device, int num_teams, class Tag>
//using SnapAoSoATeamPolicy = typename Kokkos::TeamPolicy<Device, Kokkos::LaunchBounds<vector_length * num_teams>, Tag>;

template<class NeighborClass>
ForceSNAP<NeighborClass>::ForceSNAP(char** args, System* system_, bool half_neigh_):Force(args,system_,half_neigh_)
{

  system = system_;
  nelements = 0;

  nmax = 0;

  vector_length = 8;
  concurrent_interactions =
#if defined(KOKKOS_ENABLE_CUDA)
      std::is_same<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>::value ?
          Kokkos::DefaultExecutionSpace::concurrency()/vector_length :
#elif defined(KOKKOS_ENABLE_HIP)
      std::is_same<Kokkos::DefaultExecutionSpace,Kokkos::Experimental::HIP>::value ?
          Kokkos::DefaultExecutionSpace::concurrency()/vector_length :
#elif defined(KOKKOS_ENABLE_SYCL)
      std::is_same<Kokkos::DefaultExecutionSpace,Kokkos::Experimental::SYCL>::value ?
          Kokkos::DefaultExecutionSpace::concurrency()/vector_length :
#else
          Kokkos::DefaultExecutionSpace::concurrency();
#endif

  schedule_user = 0;
  schedule_time_guided = -1;
  schedule_time_dynamic = -1;
  ncalls_neigh =-1;

  ilistmask_max = 0;
  ghostinum = 0;
  ghostilist_max = 0;
  ghostnumneigh_max = 0;
  ghostneighs_total = 0;
  ghostneighs_max = 0;

  i_max = 0;
  i_neighmax = 0;
  i_numpairs = 0;

  use_shared_arrays = 0;

#ifdef TIMING_INFO
  timers[0] = 0;
  timers[1] = 0;
  timers[2] = 0;
  timers[3] = 0;
#endif

  // Need to set this because restart not handled by ForceHybrid

  cutsq = t_fparams("ForceSNAP::cutsq",system->ntypes,system->ntypes);
}

/* ---------------------------------------------------------------------- */

template<class NeighborClass>
ForceSNAP<NeighborClass>::~ForceSNAP()
{
  // Need to set this because restart not handled by ForceHybrid

  /*if (sna) {
    for (int tid = 0; tid<concurrent_interactions; tid++)
      delete sna[tid];
    delete [] sna;

  }*/
}

template<class NeighList>
struct FindMaxNumNeighs {
  NeighList neigh_list;

  FindMaxNumNeighs(NeighList& nl): neigh_list(nl) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i, int& max_neighs) const {
    typename NeighList::t_neighs neighs_i = neigh_list.get_neighs(i);
    const int num_neighs = neighs_i.get_num_neighs();
    if(max_neighs<num_neighs) max_neighs = num_neighs;
  }
};
/* ----------------------------------------------------------------------
   This version is a straightforward implementation
   ---------------------------------------------------------------------- */

// Functor that will be replaced by LAMMPS version

template<class NeighborClass>
void ForceSNAP<NeighborClass>::compute(System* system, Binning* binning, Neighbor* neighbor_)
{

  if(comm_newton == false)
    Kokkos::abort("ForceSNAP requires 'newton on'");
  x = system->x;
  f = system->f;
  type = system->type;
  int nlocal = system->N_local;

  //class SNA* snaptr = sna[0];

  NeighborClass* neighbor = (NeighborClass*) neighbor_;
  neigh_list = neighbor->get_neigh_list();
  int max_neighs = 0;
  /*
  for (int i = 0; i < nlocal; i++) {
    typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);
    const int num_neighs = neighs_i.get_num_neighs();
    if(max_neighs<num_neighs) max_neighs = num_neighs;
  }*/
  Kokkos::parallel_reduce("ForceSNAP::find_max_neighs",nlocal, FindMaxNumNeighs<t_neigh_list>(neigh_list), Kokkos::Max<int>(max_neighs));
  // snaKK (line 220 in lammps)
  sna.nmax = max_neighs;
  // Put LAMMPS code starting here
  T_INT team_scratch_size = sna.size_team_scratch_arrays();
  T_INT thread_scratch_size = sna.size_thread_scratch_arrays();

  //printf("Sizes: %i %i\n",team_scratch_size/1024,thread_scratch_size/1024);
  int vector_length = 8;
  int team_size_max = Kokkos::TeamPolicy<>(nlocal,Kokkos::AUTO).team_size_max(*this,Kokkos::ParallelForTag());
#ifdef EMD_ENABLE_GPU
  // Same as 207 in lammps
  int team_size = 20;//max_neighs;
  if(team_size*vector_length > team_size_max)
    team_size = team_size_max/vector_length;
#else
  int team_size = 1;
#endif
      //ComputeNeigh -- lammps
      {
        // team_size_compute_neigh is defined in `pair_snap_kokkos.h`
        int scratch_size = scratch_size_helper<int>(team_size_compute_neigh * max_neighs);

        SnapAoSoATeamPolicy<team_size_compute_neigh, TagComputeNeigh> policy_neigh(chunk_size,team_size_compute_neigh,vector_length);
        policy_neigh = policy_neigh.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
        Kokkos::parallel_for("ComputeNeigh",policy_neigh,*this);
      }    



/*
  Kokkos::TeamPolicy<> policy(nlocal,team_size,vector_length);
  // Replaced in LAMMPS by a lot of functors
  Kokkos::parallel_for("ForceSNAP::compute",policy
      .set_scratch_size(1,Kokkos::PerThread(thread_scratch_size))
      .set_scratch_size(1,Kokkos::PerTeam(team_scratch_size))
    ,*this);
*/
//static int step =0;
//step++;
//if(step%10==0)
//        printf(" %e %e %e %e %e (%e %e): %e\n",t1,t2,t3,t4,t5,t6,t7,t1+t2+t3+t4+t5);
}



/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class NeighborClass>
void ForceSNAP<NeighborClass>::allocate()
{
  map = Kokkos::View<T_INT*>("ForceSNAP::map",nelements+1);
}


/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class NeighborClass>
void ForceSNAP<NeighborClass>::init_coeff(int narg, char **arg)
{
  // read SNAP element names between 2 filenames
  // nelements = # of SNAP elements
  // elements = list of unique element names

  if (narg < 7) Kokkos::abort("SNAP 1: Incorrect args for pair coefficients");
  allocate();

  if (nelements) {
    for (int i = 0; i < nelements; i++)
      delete[] elements[i];
    delete[] elements;
    radelem = Kokkos::View<double*>();
    wjelem = Kokkos::View<double*>();
    coeffelem = Kokkos::View<double**, Kokkos::LayoutRight>();
  }

  nelements = narg - 5 - system->ntypes;
  if (nelements < 1) Kokkos::abort("SNAP 2: Incorrect args for pair coefficients");

  map = Kokkos::View<T_INT*>("ForceSNAP::map",nelements+1);

  char* type1 = arg[1];
  char* type2 = arg[2];
  char* coefffilename = arg[3];
  char** elemlist = &arg[4];
  char* paramfilename = arg[4+nelements];
  char** elemtypes = &arg[5+nelements];

  // insure I,J args are * *
  if (strcmp(type1,"*") != 0 || strcmp(type2,"*") != 0)
    Kokkos::abort("A Incorrect args for pair coefficients");

  elements = new char*[nelements];

  for (int i = 0; i < nelements; i++) {
    char* elemname = elemlist[i];
    int n = strlen(elemname) + 1;
    elements[i] = new char[n];
    strcpy(elements[i],elemname);
  }

  // read snapcoeff and snapparam files

  read_files(coefffilename,paramfilename);

  if (!quadraticflag)
    ncoeff = ncoeffall - 1;
  else {

    // ncoeffall should be (ncoeff+2)*(ncoeff+1)/2
    // so, ncoeff = floor(sqrt(2*ncoeffall))-1

    ncoeff = sqrt(2*ncoeffall)-1;
    ncoeffq = (ncoeff*(ncoeff+1))/2;
    int ntmp = 1+ncoeff+ncoeffq;
    if (ntmp != ncoeffall) {
      printf("ncoeffall = %d ntmp = %d ncoeff = %d \n",ncoeffall,ntmp,ncoeff);
      Kokkos::abort("Incorrect SNAP coeff file");
    }
  }

  // read args that map atom types to SNAP elements
  // map[i] = which element the Ith atom type is, -1 if not mapped
  // map[0] is not used

  auto h_map = Kokkos::create_mirror_view(map);
  for (int i = 1; i <= system->ntypes; i++) {
    char* elemname = elemtypes[i-1];
    int jelem;
    for (jelem = 0; jelem < nelements; jelem++)
      if (strcmp(elemname,elements[jelem]) == 0)
	break;

    if (jelem < nelements)
      h_map[i] = jelem;
    else if (strcmp(elemname,"NULL") == 0) h_map[i] = -1;
    else Kokkos::abort("Incorrect args for pair coefficients");
  }

  Kokkos::deep_copy(map,h_map);
  // allocate memory for per OpenMP thread data which
  // is wrapped into the sna class


  sna = SNA(rfac0,twojmax,
            diagonalstyle,use_shared_arrays,
		        rmin0,switchflag,bzeroflag);
    //if (!use_shared_arrays)
  sna.grow_rij(nmax);
  sna.init();

  //printf("ncoeff = %d snancoeff = %d \n",ncoeff,sna[0]->ncoeff);
  if (ncoeff != sna.ncoeff) {
    printf("ncoeff = %d snancoeff = %d \n",ncoeff,sna.ncoeff);
    Kokkos::abort("Incorrect SNAP parameter file");
  }

  // Calculate maximum cutoff for all elements
  auto h_radelem = Kokkos::create_mirror_view(radelem);
  Kokkos::deep_copy(h_radelem,radelem);
  rcutmax = 0.0;
  for (int ielem = 0; ielem < nelements; ielem++) {
    rcutmax = MAX(2.0*h_radelem[ielem]*rcutfac,rcutmax);
  }
  Kokkos::deep_copy(cutsq,rcutmax*rcutmax);
  rnd_cutsq = cutsq;
}

/* ---------------------------------------------------------------------- */

template<class NeighborClass>
void ForceSNAP<NeighborClass>::read_files(char *coefffilename, char *paramfilename)
{

  // open SNAP coefficient file on proc 0

  FILE *fpcoeff;
  //if (comm->me == 0) {
    fpcoeff = fopen(coefffilename,"r");
    if (fpcoeff == NULL) {
      char str[128];
      sprintf(str,"Cannot open SNAP coefficient file %s",coefffilename);
      //error->one(FLERR,str);
    }
  //}

  char line[MAXLINE],*ptr;
  int eof = 0;

  int n;
  int nwords = 0;
  while (nwords == 0) {
    //if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpcoeff);
      if (ptr == NULL) {
        eof = 1;
        fclose(fpcoeff);
      } else n = strlen(line) + 1;
    // }
    //MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    //MPI_Bcast(&n,1,MPI_INT,0,world);
    //MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank

    if ((ptr = strchr(line,'#'))) {*ptr = '\0';}
    else if(line[0]!=10) {nwords = 2;}
    //nwords = atom->count_words(line);
  }
  if (nwords != 2)
    Kokkos::abort("Incorrect format in SNAP coefficient file");

  // words = ptrs to all words in line
  // strip single and double quotes from words
  char* words[MAXWORD];
  int iword = 0;
  words[iword] = strtok(line,"' \t\n\r\f");
  iword = 1;
  words[iword] = strtok(NULL,"' \t\n\r\f");

  int nelemfile = atoi(words[0]);
  ncoeffall = atoi(words[1]);

  // Set up element lists

  radelem = Kokkos::View<double*>("pair:radelem",nelements);
  wjelem = Kokkos::View<double*>("pair:wjelem",nelements);
  coeffelem = Kokkos::View<double**, Kokkos::LayoutRight>("pair:coeffelem",nelements,ncoeffall);

  int *found = new int[nelements];
  for (int ielem = 0; ielem < nelements; ielem++)
    found[ielem] = 0;

  // Loop over elements in the SNAP coefficient file

  for (int ielemfile = 0; ielemfile < nelemfile; ielemfile++) {

    //if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpcoeff);
      if (ptr == NULL) {
	eof = 1;
	fclose(fpcoeff);
      } else n = strlen(line) + 1;
    //}
    //MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof)
      Kokkos::abort("Incorrect format in SNAP coefficient file");
    //MPI_Bcast(&n,1,MPI_INT,0,world);
    //MPI_Bcast(line,n,MPI_CHAR,0,world);

    //nwords = atom->count_words(line);
    //if (nwords != 3)
    //  Kokkos::abort("Incorrect format in SNAP coefficient file");

    iword = 0;
    words[iword] = strtok(line,"' \t\n\r\f");
    iword = 1;
    words[iword] = strtok(NULL,"' \t\n\r\f");
    iword = 2;
    words[iword] = strtok(NULL,"' \t\n\r\f");

    char* elemtmp = words[0];
    double radtmp = atof(words[1]);
    double wjtmp = atof(words[2]);

    // skip if element name isn't in element list

    int ielem;
    for (ielem = 0; ielem < nelements; ielem++)
      if (strcmp(elemtmp,elements[ielem]) == 0) break;
    if (ielem == nelements) {
      //if (comm->me == 0)
	for (int icoeff = 0; icoeff < ncoeffall; icoeff++)
	  ptr = fgets(line,MAXLINE,fpcoeff);
      continue;
    }

    // skip if element already appeared

    if (found[ielem]) {
    //  if (comm->me == 0)
	for (int icoeff = 0; icoeff < ncoeffall; icoeff++)
	  ptr = fgets(line,MAXLINE,fpcoeff);
      continue;
    }

    found[ielem] = 1;
    auto radelem_i = Kokkos::subview(radelem,ielem);
    Kokkos::deep_copy(radelem,radtmp);
    auto wjelem_i = Kokkos::subview(wjelem,ielem);
    Kokkos::deep_copy(wjelem,wjtmp);
//    radelem[ielem] = radtmp;
//    wjelem[ielem] = wjtmp;


    //if (comm->me == 0) {
      //if (logfile) fprintf(logfile,"SNAP Element = %s, Radius %g, Weight %g \n",
			//  elements[ielem], radelem[ielem], wjelem[ielem]);
    //}

    for (int icoeff = 0; icoeff < ncoeffall; icoeff++) {
      //if (comm->me == 0) {
	ptr = fgets(line,MAXLINE,fpcoeff);
	if (ptr == NULL) {
	  eof = 1;
	  fclose(fpcoeff);
	} else n = strlen(line) + 1;
      //}

      //MPI_Bcast(&eof,1,MPI_INT,0,world);
      if (eof)
	Kokkos::abort("Incorrect format in SNAP coefficient file");
      //MPI_Bcast(&n,1,MPI_INT,0,world);
      //MPI_Bcast(line,n,MPI_CHAR,0,world);

      //nwords = atom->count_words(line);
      //if (nwords != 1)
	//Kokkos::abort("Incorrect format in SNAP coefficient file");

      iword = 0;
      words[iword] = strtok(line,"' \t\n\r\f");

      //coeffelem(ielem,icoeff) = atof(words[0]);
      auto coeffelem_ii = Kokkos::subview(coeffelem,ielem,icoeff);
      Kokkos::deep_copy(coeffelem_ii,atof(words[0]));

    }
  }

  // set flags for required keywords

  rcutfacflag = 0;
  twojmaxflag = 0;

  // Set defaults for optional keywords

  rfac0 = 0.99363;
  rmin0 = 0.0;
  diagonalstyle = 3;
  switchflag = 1;
  bzeroflag = 1;
  quadraticflag = 0;

  // open SNAP parameter file on proc 0

  FILE *fpparam;
  //if (comm->me == 0) {
    fpparam = fopen(paramfilename,"r");
    if (fpparam == NULL) {
      char str[128];
      sprintf(str,"Cannot open SNAP parameter file %s",paramfilename);
      //error->one(FLERR,str);
    }
  //}

  eof = 0;
  while (1) {
    //if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpparam);
      if (ptr == NULL) {
        eof = 1;
        fclose(fpparam);
      } else n = strlen(line) + 1;
    //}
    //MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    //MPI_Bcast(&n,1,MPI_INT,0,world);
    //MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank

    if ((ptr = strchr(line,'#'))) {*ptr = '\0'; continue;}
    //nwords = atom->count_words(line);
    if(line[0]!=10) nwords = 2; else nwords = 0;
    if (nwords == 0) continue;

    if (nwords != 2)
      Kokkos::abort("Incorrect format in SNAP parameter file");

    // words = ptrs to all words in line
    // strip single and double quotes from words

    char* keywd = strtok(line,"' \t\n\r\f");
    char* keyval = strtok(NULL,"' \t\n\r\f");

    //if (comm->me == 0) {
      //if (screen)
      //if (logfile) fprintf(logfile,"SNAP keyword %s %s \n",keywd,keyval);
    //}

    if (strcmp(keywd,"rcutfac") == 0) {
      rcutfac = atof(keyval);
      rcutfacflag = 1;
    } else if (strcmp(keywd,"twojmax") == 0) {
      twojmax = atoi(keyval);
      twojmaxflag = 1;
    } else if (strcmp(keywd,"rfac0") == 0)
      rfac0 = atof(keyval);
    else if (strcmp(keywd,"rmin0") == 0)
      rmin0 = atof(keyval);
    else if (strcmp(keywd,"diagonalstyle") == 0)
      diagonalstyle = atoi(keyval);
    else if (strcmp(keywd,"switchflag") == 0)
      switchflag = atoi(keyval);
    else if (strcmp(keywd,"bzeroflag") == 0)
      bzeroflag = atoi(keyval);
    else if (strcmp(keywd,"quadraticflag") == 0)
      quadraticflag = atoi(keyval);
    else
      Kokkos::abort("Incorrect SNAP parameter file");
  }

  if (rcutfacflag == 0 || twojmaxflag == 0)
    Kokkos::abort("Incorrect SNAP parameter file");

  delete[] found;
}

/*
template<class NeighborClass>
KOKKOS_INLINE_FUNCTION
void ForceSNAP<NeighborClass>::operator() (TagComputeNeighCPU,const typename Kokkos::TeamPolicy<TagComputeNeighCPU>::member_type& team) const {


  int ii = team.league_rank();
  const int i = d_ilist[ii + chunk_offset];
  // TODO: snaKK is used in lammps
  SNAKokkos my_sna = snaKK;
  const double xtmp = x(i,0);
  const double ytmp = x(i,1);
  const double ztmp = x(i,2);
  const int itype = type[i];
  const int ielem = d_map[itype];
  const double radi = d_radelem[ielem];

  const int num_neighs = d_numneigh[i];

  // rij[][3] = displacements between atom I and those neighbors
  // inside = indices of neighbors of I within cutoff
  // wj = weights for neighbors of I within cutoff
  // rcutij = cutoffs for neighbors of I within cutoff
  // note Rij sign convention => dU/dRij = dU/dRj = -dU/dRi

  int ninside = 0;
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,num_neighs),
      [&] (const int jj, int& count) {
    Kokkos::single(Kokkos::PerThread(team), [&] () {
      T_INT j = d_neighbors(i,jj);
      const F_FLOAT dx = x(j,0) - xtmp;
      const F_FLOAT dy = x(j,1) - ytmp;
      const F_FLOAT dz = x(j,2) - ztmp;

      const int jtype = type(j);
      const F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

      if (rsq < rnd_cutsq(itype,jtype))
       count++;
    });
  },ninside);

  d_ninside(ii) = ninside;

  if (team.team_rank() == 0)
  Kokkos::parallel_scan(Kokkos::ThreadVectorRange(team,num_neighs),
      [&] (const int jj, int& offset, bool final) {
  //for (int jj = 0; jj < num_neighs; jj++) {
    T_INT j = d_neighbors(i,jj);
    const F_FLOAT dx = x(j,0) - xtmp;
    const F_FLOAT dy = x(j,1) - ytmp;
    const F_FLOAT dz = x(j,2) - ztmp;

    const int jtype = type(j);
    const F_FLOAT rsq = dx*dx + dy*dy + dz*dz;
    const int elem_j = d_map[jtype];

    if (rsq < rnd_cutsq(itype,jtype)) {
      if (final) {
        my_sna.rij(ii,offset,0) = static_cast<real_type>(dx);
        my_sna.rij(ii,offset,1) = static_cast<real_type>(dy);
        my_sna.rij(ii,offset,2) = static_cast<real_type>(dz);
        my_sna.wj(ii,offset) = static_cast<real_type>(d_wjelem[elem_j]);
        my_sna.rcutij(ii,offset) = static_cast<real_type>((radi + d_radelem[elem_j])*rcutfac);
        my_sna.inside(ii,offset) = j;
        if (chemflag)
          my_sna.element(ii,offset) = elem_j;
        else
          my_sna.element(ii,offset) = 0;
      }
      offset++;
    }
  });
}
*/
// lammps splice

template<class NeighborClass>
KOKKOS_INLINE_FUNCTION
void ForceSNAP<NeighborClass>::operator() (TagComputeNeigh,const typename Kokkos::TeamPolicy<TagComputeNeigh>::member_type& team) const {

  SNAKokkos my_sna = snaKK;

  // extract atom number
  int ii = team.team_rank() + team.league_rank() * team.team_size();
  if (ii >= chunk_size) return;

  // get a pointer to scratch memory
  // This is used to cache whether or not an atom is within the cutoff.
  // If it is, type_cache is assigned to the atom type.
  // If it's not, it's assigned to -1.
  const int tile_size = max_neighs; // number of elements per thread
  const int team_rank = team.team_rank();
  const int scratch_shift = team_rank * tile_size; // offset into pointer for entire team
  int* type_cache = (int*)team.team_shmem().get_shmem(team.team_size() * tile_size * sizeof(int), 0) + scratch_shift;

  // Load various info about myself up front
  const int i = d_ilist[ii + chunk_offset];
  const F_FLOAT xtmp = x(i,0);
  const F_FLOAT ytmp = x(i,1);
  const F_FLOAT ztmp = x(i,2);
  const int itype = type[i];
  const int ielem = d_map[itype];
  const double radi = d_radelem[ielem];

  const int num_neighs = d_numneigh[i];

  // rij[][3] = displacements between atom I and those neighbors
  // inside = indices of neighbors of I within cutoff
  // wj = weights for neighbors of I within cutoff
  // rcutij = cutoffs for neighbors of I within cutoff
  // note Rij sign convention => dU/dRij = dU/dRj = -dU/dRi

  // Compute the number of neighbors, store rsq
  int ninside = 0;
  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,num_neighs),
    [&] (const int jj, int& count) {
    T_INT j = d_neighbors(i,jj);
    const F_FLOAT dx = x(j,0) - xtmp;
    const F_FLOAT dy = x(j,1) - ytmp;
    const F_FLOAT dz = x(j,2) - ztmp;

    int jtype = type(j);
    const F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    if (rsq >= rnd_cutsq(itype,jtype)) {
      jtype = -1; // use -1 to signal it's outside the radius
    }

    type_cache[jj] = jtype;

    if (jtype >= 0)
     count++;
  }, ninside);

  d_ninside(ii) = ninside;

  Kokkos::parallel_scan(Kokkos::ThreadVectorRange(team,num_neighs),
    [&] (const int jj, int& offset, bool final) {

    const int jtype = type_cache[jj];

    if (jtype >= 0) {
      if (final) {
        T_INT j = d_neighbors(i,jj);
        const F_FLOAT dx = x(j,0) - xtmp;
        const F_FLOAT dy = x(j,1) - ytmp;
        const F_FLOAT dz = x(j,2) - ztmp;
        const int elem_j = d_map[jtype];
        my_sna.rij(ii,offset,0) = dx;
        my_sna.rij(ii,offset,1) = dy;
        my_sna.rij(ii,offset,2) = dz;
        my_sna.wj(ii,offset) = d_wjelem[elem_j];
        my_sna.rcutij(ii,offset) = (radi + d_radelem[elem_j])*rcutfac;
        my_sna.inside(ii,offset) = j;
        if (chemflag)
          my_sna.element(ii,offset) = elem_j;
        else
          my_sna.element(ii,offset) = 0;
      }
      offset++;
    }
  });
}


/*
// This is multiple smaller functors in lammps
template<class NeighborClass>
KOKKOS_INLINE_FUNCTION
void ForceSNAP<NeighborClass>::operator() (const Kokkos::TeamPolicy<>::member_type& team) const {
  const int i = team.league_rank();
  SNA my_sna(sna,team);
  const double x_i = x(i,0);
  const double y_i = x(i,1);
  const double z_i = x(i,2);
  const int type_i = type[i];
  const int elem_i = map[type_i];
  const double radi = radelem[elem_i];

  typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);

  const int num_neighs = neighs_i.get_num_neighs();

  // rij[][3] = displacements between atom I and those neighbors
  // inside = indices of neighbors of I within cutoff
  // wj = weights for neighbors of I within cutoff
  // rcutij = cutoffs for neighbors of I within cutoff
  // note Rij sign convention => dU/dRij = dU/dRj = -dU/dRi

  //Kokkos::Timer timer;
  int ninside = 0;
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,num_neighs),
      [&] (const int jj, int& count) {
    Kokkos::single(Kokkos::PerThread(team), [&] (){
      T_INT j = neighs_i(jj);
      const double dx = x(j,0) - x_i;
      const double dy = x(j,1) - y_i;
      const double dz = x(j,2) - z_i;

      const int type_j = type(j);
      const double rsq = dx*dx + dy*dy + dz*dz;
      const int elem_j = map[type_j];

      if( rsq < rnd_cutsq(type_i,type_j) )
       count++;
    });
  },ninside);

  //t1 += timer.seconds(); timer.reset();

  if(team.team_rank() == 0)
  Kokkos::parallel_scan(Kokkos::ThreadVectorRange(team,num_neighs),
      [&] (const int jj, int& offset, bool final){
  //for (int jj = 0; jj < num_neighs; jj++) {
    T_INT j = neighs_i(jj);
    const double dx = x(j,0) - x_i;
    const double dy = x(j,1) - y_i;
    const double dz = x(j,2) - z_i;

    const int type_j = type(j);
    const double rsq = dx*dx + dy*dy + dz*dz;
    const int elem_j = map[type_j];

    if( rsq < rnd_cutsq(type_i,type_j) ) {
      if(final) {
        my_sna.rij(offset,0) = dx;
        my_sna.rij(offset,1) = dy;
        my_sna.rij(offset,2) = dz;
        my_sna.inside[offset] = j;
        my_sna.wj[offset] = wjelem[elem_j];
        my_sna.rcutij[offset] = (radi + radelem[elem_j])*rcutfac;
      }
      offset++;
    }
  });

  //t2 += timer.seconds(); timer.reset();

  team.team_barrier();
  // compute Ui, Zi, and Bi for atom I
  my_sna.compute_ui(team,ninside);
  //t3 += timer.seconds(); timer.reset();
  team.team_barrier();
  my_sna.compute_zi(team);
  //t4 += timer.seconds(); timer.reset();
  team.team_barrier();

  // for neighbors of I within cutoff:
  // compute dUi/drj and dBi/drj
  // Fij = dEi/dRj = -dEi/dRi => add to Fi, subtract from Fj

  Kokkos::View<double*,Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Unmanaged>>
    coeffi(coeffelem,elem_i,Kokkos::ALL);

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team,ninside),
      [&] (const int jj) {
  //for (int jj = 0; jj < ninside; jj++) {
    int j = my_sna.inside[jj];
    //Kokkos::Timer timer2;
    my_sna.compute_duidrj(team,&my_sna.rij(jj,0),
                           my_sna.wj[jj],my_sna.rcutij[jj]);
    //t6 += timer2.seconds(); timer2.reset();
    my_sna.compute_dbidrj(team);
    //t7 += timer2.seconds(); timer2.reset();
    my_sna.copy_dbi2dbvec(team);


    Kokkos::single(Kokkos::PerThread(team), [&] (){
    double fij[3];

    fij[0] = 0.0;
    fij[1] = 0.0;
    fij[2] = 0.0;

    // linear contributions

    for (int k = 1; k <= ncoeff; k++) {
      double bgb = coeffi[k];
      fij[0] += bgb*my_sna.dbvec(k-1,0);
      fij[1] += bgb*my_sna.dbvec(k-1,1);
      fij[2] += bgb*my_sna.dbvec(k-1,2);
    }

    const double dx = my_sna.rij(jj,0);
    const double dy = my_sna.rij(jj,1);
    const double dz = my_sna.rij(jj,2);
    const double fdivr = -1.5e6/pow(dx*dx + dy*dy + dz*dz,7.0);
    fij[0] += dx*fdivr;
    fij[1] += dy*fdivr;
    fij[2] += dz*fdivr;

    //OK
    //printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf SNAP-COMPARE: FIJ\n"
    //    ,x(i,0),x(i,1),x(i,2),x(j,0),x(j,1),x(j,2),fij[0],fij[1],fij[2] );
    f(i,0) += fij[0];
    f(i,1) += fij[1];
    f(i,2) += fij[2];
    f(j,0) -= fij[0];
    f(j,1) -= fij[1];
    f(j,2) -= fij[2];
    });
  });
  //t5 += timer.seconds(); timer.reset();
}
*/


template<class NeighborClass>
template<typename scratch_type>
int ForceSNAP<NeighborClass>::scratch_size_helper(int values_per_team) {
  typedef Kokkos::View<scratch_type*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewType;

  return ScratchViewType::shmem_size(values_per_team);
}

