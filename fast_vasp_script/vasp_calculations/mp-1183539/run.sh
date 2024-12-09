#!/bin/sh
#$ -cwd
# Uses one node of O-type
#$ -l cpu_40=1
#$ -l h_rt=10:00:00
#$ -N mp-1183539
# pass to the VASP executable file
PRG=/gs/fs/tga-ishikawalab/vasp/vasp.6.4.3/bin/vasp_std

. /etc/profile.d/modules.sh
module load cuda
module load intel
# Loading Intel MPI
module load intel-mpi

# Uses 8 processes with MPI
mpiexec.hydra -ppn 8 -n 8 ${PRG} >& vasp.out