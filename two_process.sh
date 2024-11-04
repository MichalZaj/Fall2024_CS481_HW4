#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn

module load openmpi/4.1.4-gcc11

mpicc -g -Wall -std=c99 -o game_of_life_mpi game_of_life_mpi.c

mpirun -n 2 ./game_of_life_mpi 5000 5000 2 /scratch/$USER
mpirun -n 2 ./game_of_life_mpi 5000 5000 2 /scratch/$USER
mpirun -n 2 ./game_of_life_mpi 5000 5000 2 /scratch/$USER
