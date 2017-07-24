# Demonstrate SNAP Ta potential

# Initialize simulation

units		metal

# generate the box and atom positions using a BCC lattice

#boundary	p p p

lattice         sc 3.1803
region		box block 0 4 0 4 0 4
create_box	1 box
create_atoms	1 box

mass 1 183.84

# SNAP W without pair hybrid zbl

pair_style snap optimized 0
pair_coeff * * W.snapcoeff W W.snapparam W

# Check per-atom virial

#compute         patom all stress/atom NULL
#compute         psum all reduce sum c_patom[1] c_patom[2] c_patom[3]
#variable 	patoms equal -(c_psum[1]+c_psum[2]+c_psum[3])/(3.0*vol)

# Setup output

thermo		10
#thermo_style custom step temp epair emol etotal press v_patoms
#thermo_modify norm yes

# Set up NVE run

#timestep 0.5e-3
neighbor 1.0 bin
neigh_modify once no every 1 delay 0 check yes

# Run MD

velocity all create 300.0 4928459 loop geom
fix 1 all nve
run            1000


