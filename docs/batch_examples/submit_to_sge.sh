#!/bin/sh
#An example script for submiting a GPI-2 test (proc_init.bin) to
#a SGE's queue using one process per node and enabling NUMA.
#It is assumed the mpi PE is setup with a `round_robin` allocation rule.

#$ -S /bin/sh
#$ -N gpi_test
#$ -pe mpi 2
#$ -V
#$ -cwd
#$ -w e

gaspihome=$HOME/localgpi/ethernet_gcc
gaspirun=$gaspihome/bin/gaspi_run

# Generate a `machine` file from $PE_HOSTFILE. Repeated hosts are
# omitted.
machine_file="machines_"$JOB_ID
cat $PE_HOSTFILE | cut -f1 -d" " | cut -f1 -d. >> $machine_file

test_bin="proc_init.bin"
$gaspirun -m $machine_file -N $gaspihome/tests/bin/$test_bin
