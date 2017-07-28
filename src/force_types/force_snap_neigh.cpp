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
#include "math_const.h"

#define MAXLINE 1024
#define MAXWORD 3

// Outstanding issues with quadratic term
// 1. there seems to a problem with compute_optimized energy calc
// it does not match compute_regular, even when quadratic coeffs = 0

/* ---------------------------------------------------------------------- */

ForceSNAP::ForceSNAP(char** args, System* system_, bool half_neigh_):Force(args,system_,half_neigh_) 
{

  system = system_;
  nelements = 0;

  nmax = 0;

  vector_length = 8;
  concurrent_interactions =
#if defined(KOKKOS_ENABLE_CUDA)
      std::is_same<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>::value ?
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

ForceSNAP::~ForceSNAP()
{
  // Need to set this because restart not handled by ForceHybrid

  /*if (sna) {
    for (int tid = 0; tid<concurrent_interactions; tid++)
      delete sna[tid];
    delete [] sna;

  }*/
}

/* ----------------------------------------------------------------------
   This version is a straightforward implementation
   ---------------------------------------------------------------------- */

void ForceSNAP::compute(System* system, Binning* binning, Neighbor* neighbor_)
{

  x = system->x;
  f = system->f;
  type = system->type;
  int nlocal = system->N_local;

  //class SNA* snaptr = sna[0];

  NeighborCSR<t_neigh_mem_space>* neighbor = (NeighborCSR<t_neigh_mem_space>*) neighbor_;
  neigh_list = neighbor->get_neigh_list();
  int max_neighs = 0;

  for (int i = 0; i < nlocal; i++) {
    typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);
    const int num_neighs = neighs_i.get_num_neighs();
    if(max_neighs<num_neighs) max_neighs = num_neighs;
  }

  sna.nmax = max_neighs;

  T_INT team_scratch_size = sna.size_team_scratch_arrays();
  T_INT thread_scratch_size = sna.size_thread_scratch_arrays();

  //printf("Sizes: %i %i\n",team_scratch_size/1024,thread_scratch_size/1024);
#ifdef KOKKOS_ENABLE_CUDA
  int team_size = max_neighs;
#else
  int team_size = 1;
#endif
  Kokkos::TeamPolicy<> policy(nlocal,team_size,4);
  Kokkos::parallel_for(policy
      .set_scratch_size(1,Kokkos::PerThread(thread_scratch_size))
      .set_scratch_size(1,Kokkos::PerTeam(team_scratch_size))
    ,*this);
}



/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void ForceSNAP::allocate()
{
  map = Kokkos::View<T_INT*>("ForceSNAP::map",nelements+1);
}


/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void ForceSNAP::init_coeff(int narg, char **arg)
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
    radelem = Kokkos::View<T_F_FLOAT*>();
    wjelem = Kokkos::View<T_F_FLOAT*>();
    coeffelem = Kokkos::View<T_F_FLOAT**, Kokkos::LayoutRight>();
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

  for (int i = 1; i <= system->ntypes; i++) {
    char* elemname = elemtypes[i-1];
    int jelem;
    for (jelem = 0; jelem < nelements; jelem++)
      if (strcmp(elemname,elements[jelem]) == 0)
	break;

    if (jelem < nelements)
      map[i] = jelem;
    else if (strcmp(elemname,"NULL") == 0) map[i] = -1;
    else Kokkos::abort("Incorrect args for pair coefficients");
  }

  // allocate memory for per OpenMP thread data which
  // is wrapped into the sna class

//#if defined(_OPENMP)
//#pragma omp parallel default(none)
//#endif
  printf("Twojmax: %i\n",twojmax);
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

  rcutmax = 0.0;
  for (int ielem = 0; ielem < nelements; ielem++) {
    rcutmax = MAX(2.0*radelem[ielem]*rcutfac,rcutmax);
  }
  Kokkos::deep_copy(cutsq,rcutmax*rcutmax);
  rnd_cutsq = cutsq;
}

/* ---------------------------------------------------------------------- */

void ForceSNAP::read_files(char *coefffilename, char *paramfilename)
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

  radelem = Kokkos::View<T_F_FLOAT*>("pair:radelem",nelements);
  wjelem = Kokkos::View<T_F_FLOAT*>("pair:wjelem",nelements);
  coeffelem = Kokkos::View<T_F_FLOAT**, Kokkos::LayoutRight>("pair:coeffelem",nelements,ncoeffall);

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
    radelem[ielem] = radtmp;
    wjelem[ielem] = wjtmp;


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

      coeffelem(ielem,icoeff) = atof(words[0]);

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
