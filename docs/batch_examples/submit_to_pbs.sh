#!/bin/sh
#An example script for submiting a GPI-2 test (proc_init.bin) to
#a PBS's queue using one process per node and enabling NUMA.

#PBS -l nodes=2
#PBS -l walltime=00:05:00
#PBS -S /bin/sh
#PBS -N PBS_test
#PBS -V

gaspihome=$HOME/localgpi/ethernet_gcc
gaspirun=$gaspihome/bin/gaspi_run

# Generate a `machine` file from $PBS_NODEFILE. Repeated hosts are
# omitted.
cd $PBS_O_WORKDIR
machine_file="machines_"$PBS_JOBID
uniq $PBS_NODEFILE > $machine_file

test_bin="proc_init.bin"
$gaspirun -m $machine_file -N $gaspihome/tests/bin/$test_bin
