#!/bin/sh
#An example script for submiting a GPI-2 test (proc_init.bin) to
#a SLURM's queue using one process per node and enabling NUMA.

#SBATCH --partition=mpi
#SBATCH --job-name gpi_test
#SBATCH --time=00:05:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1

gaspihome=$HOME/localgpi/ethernet_gcc
gaspirun=$gaspihome/bin/gaspi_run

#Machine file name
machine_file="machines_"$SLURM_JOB_ID
if [ -e $machine_file ];then
    rm -f $machine_file
fi

# Parse Slurm nodes list
# It is assumed the nodes are numbered, as it is the usual case in
# many environments, e.g.:
#  node010
#  node012
#  node120
# Repeated hosts are omitted.
nodes_base_name=`echo $SLURM_JOB_NODELIST | cut -f1 -d\[`
nodes_list=`echo $SLURM_JOB_NODELIST | cut -f1 -d\] | cut -f2 -d\[`

## Just one node case
if [ `echo $nodes_list | awk -F ',' '{print NF}'` = "1" ]; then
    echo $nodes_list >> $machine_file
    echo $machine_file
    exit
fi

i=1
node_host=`echo $nodes_list | cut -f$i -d,`
while [ ! -z $node_host ]; do
    # Several nodes in sequence, e.g., node[056-060]
    init_seq=`echo $node_host | cut -f1 -d-`
    end_seq=`echo $node_host | cut -f2 -d-`
    if [ $init_seq != $end_seq ]; then
	for ((host_cnt=`echo $init_seq | bc`;
	      host_cnt<=`echo $end_seq | bc`;
	      host_cnt++)); do
	    if [ ${init_seq:0:2} = "00" -a $host_cnt -lt 100 ]; then
		echo $nodes_base_name"00"$host_cnt >> $machine_file
	    elif [ ${init_seq:0:1} = "0" -a $host_cnt -lt 100 ]; then
		echo $nodes_base_name"0"$host_cnt >> $machine_file
	    else
		echo $nodes_base_name$host_cnt >> $machine_file
	    fi
	done
    # Several non sequential nodes, e.g., node010, node100
    else
	echo $nodes_base_name$node_host >> $machine_file
    fi
    i=`expr $i + 1`
    node_host=`echo $nodes_list | cut -f$i -d,`
done

test_bin="proc_init.bin"
$gaspirun -m $machinefile -N $gaspihome/tests/bin/$test_bin
