#!/bin/sh

# Copyright (c) Fraunhofer ITWM - Carsten Lojewski <lojewski@itwm.fhg.de>, 2013-2024

# This file is part of GPI-2.

# GPI-2 is free software; you can redistribute it
# and/or modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# GPI-2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GPI-2. If not, see <http://www.gnu.org/licenses/>.

SET_NUMA=0
MFILE=""
MASTER_PRG=""
PRG=""
PRG_ARGS=""
TYPE=GASPI_MASTER
MAX_NUMA_NODES=1
HAS_NUMA=0
TMP_PATTERN="/tmp/.gpi2.XXXXXXXX"
TMP_MFILE=$(mktemp ${TMP_PATTERN})
UNIQ_MFILE=$(mktemp ${TMP_PATTERN})
ORIG_MFILE=""
NTASKS=0
GASPI_LAUNCHER="srun"
PING_HOSTS=0

remove_temp_file()
{
    rm -f $TMP_MFILE
    rm -f $UNIQ_MFILE
}

#helper functions
usage()
{
    echo
    echo "Usage: gaspi_run -m <machinefile> [OPTIONS] <path GASPI program>"
    echo
    echo "Available options:"
    echo "  -m               machine file (optional)"
    echo "  -N               Enable NUMA for processes on same node."
    echo "  -n <procs>       Start as many <procs> from machine file."
    echo "  -p               Ping hosts before starting binary."
    echo "  -h               This help."
    echo
    remove_temp_file
}

clean_exit()
{
    remove_temp_file
}

print_error_exit()
{
    echo
    echo "Error: $1"
    echo
    remove_temp_file

    exit 1
}

validate_machinefile()
{
    if [ ! -s $MFILE ]; then
        print_error_exit "Empty machine file ($1)"
    fi
    # newline at the end
    endl=`tail -c 1 $MFILE`
    if [ "$endl" != "" ]; then
        print_error_exit "No newline at end of machine file ($1)";
    fi

    # create tmp_file
    rm -f $TMP_MFILE

    touch $TMP_MFILE 2>/dev/null && chmod 777 $TMP_MFILE ||
            {
                print_error_exit "User permissions failure: $PWD is not writable"
            }

    ncount=0
    while read LINE
    do
        i=`echo $LINE | gawk '{print $1}'   `
        if [ -n "$i" ]; then
            if [ $PING_HOSTS -eq 1 ]; then
                ping -c 1 $i >/dev/null 2>&1 ||
                    {
                        print_error_exit "Host not reachable ($i)"
                    }
            fi
            ncount=$((ncount+1))
            if [ $NTASKS -lt $ncount ]; then
                break
            fi
            echo $i >> $TMP_MFILE
        fi
    done < $MFILE

    ORIG_MFILE=$MFILE
    MFILE=$TMP_MFILE

    uniq $MFILE > $UNIQ_MFILE
    # number of nodes (NTASKS) must fit number of hosts
    n=$(wc -l < $MFILE)
    if [ $n -lt $NTASKS ]; then
        print_error_exit "Not enough hosts ($n) for required number of nodes (-n $NTASKS)"
    fi
}

#at least we need the binary
if [ $# -lt 1 ]; then
    usage
    exit 1
fi

#command line parsing
while [ -n "$1" ]; do
    case $1 in
        -m | --machinefile)
            shift
            if [ -r $1 ]; then
                MFILE=$1
            else
                print_error_exit "Cannot read $1 (-m option) (or file does not exist)"
            fi
            ;;
        -N | --NUMA)
            SET_NUMA=1
            ;;
        -h | --help)
            usage
            ;;
        -n | --nodes)
            shift
            NTASKS=$1
            ;;
        -p | --ping)
            PING_HOSTS=1
            ;;
        -*)
            print_error_exit "Invalid option ($1)"
            ;;
        *) #Sucks! we're taking a small risk here
            if [ -x $1 ]; then

                PRG=$(readlink -f $1)
                shift
                PRG_ARGS="$*"
                break
            else
                print_error_exit "Cannot execute $1 (or file does not exist)"
            fi
    esac
    shift
done


if [ -z "$PRG" ]; then
    print_error_exit "No binary file provided. See help (-h option)"
fi

#set machine file
if [ -z "$MFILE" ]; then
    MFILE="machines_"$SLURM_JOB_ID
    scontrol show hostnames > $MFILE
fi

#sanity check and settings
if [ $SET_NUMA -eq 1 ]; then
    which lscpu > /dev/null
    if [ $? = 0 ]; then
        MAX_NUMA_NODES=`lscpu|grep "NUMA node(s)"|gawk '{print $3}'`
        HAS_NUMA=1
    else
        echo "Warning: unable to detect NUMA configuration. Option not used."
        MAX_NUMA_NODES=1
        HAS_NUMA=0
    fi
else
    MAX_NUMA_NODES=256
fi

#let's rock
#use all hosts in machines file
if [ $NTASKS -eq 0 ]; then
    NTASKS=`sed /^$/d $MFILE  |wc -l`
fi

validate_machinefile $MFILE

if [ $NTASKS -gt $SLURM_NPROCS ]; then
    print_error_exit "Number of hosts in machinefile (${NTASKS}) exceeds the job's allocation number (${SLURM_NPROCS})."
fi

#master binary is the same
if [ -z "$MASTER_PRG" ]; then
    MASTER_PRG=$PRG
fi

if [ 0 -eq "$SLURM_PROCID" ]; then
    NNODES=`wc -l < $UNIQ_MFILE`
fi

if [ $(($NTASKS / $NNODES)) -gt $MAX_NUMA_NODES ]; then
    print_error_exit "NUMA option: incorrect machinefile (-m $ORIG_MFILE) (max procs per node: $MAX_NUMA_NODES)"
fi

$GASPI_LAUNCHER -n $NTASKS -N $NNODES -w $MFILE $(readlink -f `dirname $0`)/slurm.env $MFILE $SET_NUMA $MASTER_PRG $PRG_ARGS


if [ $? != 0 ]; then
    clean_exit 1
    exit 1
fi

wait

#clean-up and exit
clean_exit 0

exit 0
