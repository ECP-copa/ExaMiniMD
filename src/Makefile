#Kokkos root path
KOKKOS_PATH = ${HOME}/kokkos

#Backend and architecture configuration
KOKKOS_DEVICES=OpenMP
KOKKOS_ARCH = "SNB,Kepler35"

#MPI On or off (1/0)
MPI = 1


default: build
	echo "Start Build"


EXE = ExaMiniMD
CXXFLAGS = -O3 -g
LINKFLAGS = -O3 -g

ifeq ($(MPI), 1)
  CXX = mpicxx
  CXXFLAGS += -DEXAMINIMD_ENABLE_MPI
else
  ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
    CXX = $(KOKKOS_PATH)/bin/nvcc_wrapper
  else
    CXX = g++
  endif
endif

LINK = ${CXX}

KOKKOS_OPTIONS=aggressive_vectorization

DEPFLAGS = -M

SRC = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o $(EXE)

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) ${HEADERS}
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

