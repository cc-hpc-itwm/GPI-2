#!/bin/sh
#An example script for submiting a GPI-2 test (proc_init.bin) to
#a SLURM's queue using one process per node and enabling NUMA.

#SBATCH --partition=mpi
#SBATCH --job-name gpi_test
#SBATCH --time=00:05:00
#SBATCH --nodes=11
#SBATCH --ntasks-per-node=1

gaspihome=$HOME/local
gaspirun=$gaspihome/bin/gaspi_run

#Machine file name
machine_file="machines_"$SLURM_JOB_ID
if [ -e $machine_file ];then
    rm -f $machine_file
fi

scontrol show hostnames > $machine_file

test_bin="proc_init.bin"
$gaspirun -m $machine_file -N $gaspihome/tests/bin/$test_bin
