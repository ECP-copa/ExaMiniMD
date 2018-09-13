# Demonstrate SNAP Ta potential

# Initialize simulation

units		metal

newton 		on
# generate the box and atom positions using a BCC lattice

lattice         sc 3.1803
region		box block 0 8 0 8 0 8
create_box	1 box
create_atoms	1 box

mass 1 183.84

# SNAP W without pair hybrid zbl

pair_style snap optimized 1
pair_coeff * * W.snapcoeff W W.snapparam W


# Setup output

thermo		10

# Set up NVE run

neighbor 1.0 bin
neigh_modify once no every 20 delay 0 check yes

# Run MD

velocity all create 300.0 4928459 loop geom
fix 1 all nve
run            100


