#!/bin/sh

GPI2_PATH=/opt/GPI2
OFED_PATH=""
OFED=0
WITH_MPI=0
MPI_PATH=""
WITH_LL=0
WITH_F90=1
WITH_CUDA=0
CUDA_PATH=""

usage()
{
cat << EOF

GPI2 Installation:

    Usage: `basename $0` [-p PATH_GPI2_INSTALL] [-o OFED_DIR] <further options>
      where
             -p Path where to install GPI-2 (default: ${GPI2_PATH})
             -o Path to OFED installation

    Further options:
             --with-mpi<=path>              Use this option if you aim at mixed mode with MPI.
	                                    See README for more information.
             --with-ll                      Use this option if you have Load Leveler as batch system.
	                                    This integrates with Load Leveler and uses poe as application launcher.
             --with-fortran=(true,false)    Enable/Disable Fortran bindings (default: enabled).

	     --with-cuda<=path>             Use this option if you aim mixed use with CUDA and GPU support.
	     	     
EOF
}

clean_bak_files()
{
    if [ -r src/make.inc.bak ]; then
	cp src/make.inc src/make.inc.install
	mv src/make.inc.bak src/make.inc
    fi
    
    if [ -r tests/make.defines.bak ]; then
	cp tests/make.defines tests/make.defines.install
	mv tests/make.defines.bak tests/make.defines
    fi
}

while getopts ":p:o:-:" opt; do
    case $opt in
	-)
	    case "${OPTARG}" in
		with-mpi)
		    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
		    if [ "$val" = "" ]; then
			which mpirun > /dev/null 2>&1
			if [ $? != 0 ]; then
			    echo "Couldn't find MPI installation. Please provide path to your MPI installation."
			    echo "    ./install <other options> --with-mpi=<Path to MPI installation>"
			    echo ""
			    exit 1
			fi
			MPI_BIN=`which mpirun`
			MPI_PATH=`dirname $MPI_BIN`
			MPI_PATH=`dirname $MPI_PATH`
		    else
			MPI_PATH=$val
		    fi
		    echo "With MPI at ${MPI_PATH}" >&2;
		    WITH_MPI=1
		    ;;
		with-mpi=*)
		val=${OPTARG#*=}
		if [ "$val" = "" ]; then
		    echo "Forgot to provide MPI path?"
		    exit 1
		fi
		MPI_PATH=$val
		opt=${OPTARG%=$val}
		echo "With MPI at ${MPI_PATH}" >&2;
		WITH_MPI=1
		;;
		with-ll)
		    echo "With Load Leveler" >&2;
		    WITH_LL=1
		    ;;
		with-fortran)
		    WITH_F90=1
		    ;;
		with-fortran=*)
		val=${OPTARG#*=}
		if [ "$val" == "false" ]; then
		    WITH_F90=0
		    sed -i "s,fortran,,g" tests/tests/Makefile
		fi
		;;
		with-cuda)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    if [ "$val" = "" ]; then
                        which nvcc > /dev/null 2>&1
                        if [ $? != 0 ]; then
                            echo "Couldn't find CUDA installation. Please provide path to your CUDA installation."
                            exit 1
                        fi
			NVCC_BIN=`which nvcc`
			CUDA_PATH=`dirname $NVCC_BIN`
			CUDA_PATH=`dirname $CUDA_PATH`
		    else
			CUDA_PATH=$val
                    fi
		    
                    echo "With CUDA at ${CUDA_PATH}" >&2;
                    WITH_CUDA=1
                    ;;
                with-cuda=*)
                val=${OPTARG#*=}
                if [ "$val" = "" ]; then
                    echo "Forgot to provide CUDA path?"
                    exit 1
                fi
                CUDA_PATH=$val
                opt=${OPTARG%=$val}
                echo "With CUDA at ${CUDA_PATH}" >&2;
                WITH_CUDA=1
                ;;
		*)
		    if [ "$OPTERR" = 1 ] && [ "${optspec:0:1}" != ":" ]; then
		        echo "Unknown option --${OPTARG}" >&2
			usage
			exit 1
		    fi
		    ;;
	    esac;;
	o)
	    echo "Path to OFED to be used: $OPTARG" >&2
	    OFED_PATH=$OPTARG
	    OFED=1
	    ;;
        p)
            echo "Installation path to be used: $OPTARG" >&2
            GPI2_PATH=$OPTARG
            ;;
        \?)
            echo "Unknown option: -$OPTARG" >&2
	    usage
	    exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done
 
#remove (old) log
rm -f install.log

echo "$0 $@" >> install.log

#check ofed installation
if [ $OFED = 0 ]; then
    echo "Searching OFED installation..." | tee -a install.log
    INFO=/etc/infiniband/info
    if [ -x $INFO ]; then
	
	OFED_PATH=$(${INFO} | grep -w prefix | cut -d '=' -f 2)
	echo "Found OFED installation in $OFED_PATH" | tee -a install.log
    else
	if [ -r /usr/lib64/libibverbs.so ] && [ -r /usr/include/infiniband/verbs.h ]; then
	    OFED_PATH=/usr/    
	else
	    echo "Error: could not find libibverbs."
	    echo "Run this script with the -o option and provide the path to your OFED installation."
	    echo
	    exit 1
	fi
    fi
else
    if [ ! -d $OFED_PATH ]; then
	echo "Error: $OFED_PATH not a directory"
	exit 1
    fi
fi

#check ibverbs support
grep IBV_LINK_LAYER_ETHERNET $OFED_PATH/include/infiniband/verbs.h > /dev/null
if [ $? != 0 ]; then
    echo "Error: Too old version of libibverbs (need at least v1.1.6)."
    echo "Please update your OFED stack to a more recent version."
    exit 1
fi
    
sed -i  "s,OFED_PATH = /usr,OFED_PATH = $OFED_PATH,g" src/make.inc
sed -i  "s,OFED_PATH = /usr,OFED_PATH = $OFED_PATH,g" tests/make.defines

#MPI mixed mode
if [ $WITH_MPI = 1 ]; then

    #check
    if [ -r $MPI_PATH/include64/mpi.h ]; then
	MPI_INC_PATH=$MPI_PATH/include64
    else
	if [ -r $MPI_PATH/include/mpi.h ]; then
	    MPI_INC_PATH=$MPI_PATH/include
	else
	    echo "Cannot find mpi.h. Please provide path to MPI installation."
	    echo "    ./install <other options> --with-mpi=<Path to MPI installation>"
	    echo ""
	    exit 1
	fi
    fi

    if [ -r $MPI_PATH/lib64/libmpi.so ] || [ -r $MPI_PATH/lib64/libmpi.a ]; then
	MPI_LIB_PATH=$MPI_PATH/lib64
    else
	if [ -r $MPI_PATH/lib/libmpi.so ] || [ -r $MPI_PATH/lib/libmpi.a ]; then
	    MPI_LIB_PATH=$MPI_PATH/lib
	else
	    echo "Cannot find libmpi. Please provide path to MPI installation."
	    echo "    ./install <other options> --with-mpi=<Path to MPI installation>"
	    echo ""
	    exit 1
	fi
    fi
    
    cp src/make.inc src/make.inc.bak
    echo "Using MPI: ${MPI_PATH}" | tee -a install.log
    echo "###### added by install script" >> src/make.inc
    echo "CFLAGS += -DGPI2_WITH_MPI" >> src/make.inc
    echo "DBG_CFLAGS += -DGPI2_WITH_MPI" >> src/make.inc
    echo "INCLUDES += -I${MPI_INC_PATH}" >> src/make.inc

    cp tests/make.defines tests/make.defines.bak
    echo "###### added by install script" >> tests/make.defines
    echo "CFLAGS += -DGPI2_WITH_MPI -I${MPI_INC_PATH} -L${MPI_LIB_PATH}" >> tests/make.defines
    echo "LIB_PATH += -L${MPI_LIB_PATH}" >> tests/make.defines    
    echo "LIBS += -lmpi" >> tests/make.defines
    echo "export" >> tests/make.defines

fi

#load leveler
if [ $WITH_LL = 1 ]; then
    echo "Setup for Load Leveler" | tee -a install.log
    echo "###### added by install script" >> src/make.inc
    echo "CFLAGS += -DLOADLEVELER" >> src/make.inc
    echo "DBG_CFLAGS += -DLOADLEVELER" >> src/make.inc
fi

#CUDA/GPU support 
if [ $WITH_CUDA = 1 ]; then

    #check
    #check moduel
    if [ `modprobe -l |grep nv_peer_mem` ]; then
      echo "nv_peer_mem module is found, continue"
    else
      echo "cannot find nv_peer_mem module, return"
      echo ""
      exit 1
   fi

    if [ -r $CUDA_PATH/include/cuda.h ]; then
	CUDA_INC_PATH=$CUDA_PATH/include
    else
	    echo "Cannot find cuda.h. Please provide path to CUDA installation."
	    echo "    ./install.sh <other options> --with-cuda=<Path to CUDA installation>"
	    echo ""
	    exit 1
    fi

    if [ -r $CUDA_PATH/lib64/libcudart.so ] && [ -r $CUDA_PATH/lib64/libcuinj64.so ] ; then
	CUDA_LIB_PATH=$CUDA_PATH/lib64
    else
	if [ -r $CUDA_PATH/lib/libcudart.so ] && [ -r $CUDA_PATH/li32/libcuinj32.so ] ; then
	    CUDA_LIB_PATH=$CUDA_PATH/lib
	else
	    echo "Cannot find libcudart or libcuinj. Please provide path to CUDA installation."
	    echo "    ./install.sh <other options> --with-cuda=<Path to CUDA installation>"
	    echo ""
	    exit 1
	fi
    fi
   
    cp src/make.inc src/make.inc.bak
    echo "Using CUDA: ${CUDA_PATH}" | tee -a install.log
    echo "###### added by install script" >> src/make.inc
    echo "SRCS += GPI2_GPU.c" >>src/make.inc
    echo "HDRS += GPI2_GPU.h" >>src/make.inc
    echo "CFLAGS += -DGPI2_CUDA" >> src/make.inc
    echo "DBG_CFLAGS += -DGPI2_CUDA" >> src/make.inc
    echo "INCLUDES += -I${CUDA_INC_PATH}" >> src/make.inc

    echo "###### added by install script" >> tests/make.defines
    echo "CFLAGS += -DGPI2_CUDA -I${CUDA_INC_PATH}" >> tests/make.defines
    echo "LIB_PATH += -L${CUDA_LIB_PATH}" >> tests/make.defines  
    if [ -r $CUDA_PATH/lib64/libcuinj64.so ] ; then 
	echo "LIBS += -lcudart -lcuinj64" >> tests/make.defines
    else
	echo "LIBS += -lcudart -lcuinj32" >> tests/make.defines
    fi
    echo "export" >> tests/make.defines
fi


#build everything
make clean &> /dev/null
make -C src depend &> /dev/null

echo -e -n "\nBuilding GPI..."
make gpi >> install.log 2>&1
if [ $? != 0 ]; then
    echo "Compilation of GPI-2 failed (see install.log)"
    echo "Aborting..."
    clean_bak_files
    exit -1
fi

if [ $WITH_F90 = 1 ]; then
    make fortran >> install.log 2>&1
    if [ $? != 0 ]; then
	echo "Creation of GPI-2 Fortran bindings failed (see install.log)"
	echo "Aborting..."
	clean_bak_files
	exit -1
    fi
fi
echo " done."


echo -e -n "\nBuilding tests..."
make tests >> install.log 2>&1
if [ $? != 0 ]; then
    echo "Compilation of tests failed (see install.log)"
    echo "Aborting..."
    clean_bak_files
    exit -1
fi
echo " done."

echo -e -n "\nCreating documentation..."
make docs >> install.log 2>&1
if [ $? != 0 ]; then
    echo "Failed to create documentation (see install.log)"
else
    echo " done."
fi

#copy things to the right place
echo
if [ ! -d "$GPI2_PATH" ]; then
    echo "Creating installation directory: ${GPI2_PATH}" |tee -a install.log
    mkdir -p $GPI2_PATH 2>> install.log 
    if [ "$?" != "0" ] ; then
	echo
        echo "Failed to create directory (${GPI2_PATH}). Check your permissions or choose a different directory."
	echo "You can use the (-p) option to choose a different directory."
	echo "Your installation was not completed!"
	echo
	clean_bak_files
        exit 1
    fi

fi

mkdir -p $GPI2_PATH/bin
if [ $WITH_LL = 1 ]; then
    cp bin/gaspi_run.poe $GPI2_PATH/bin/gaspi_run
    #create dummy cleanup
    head -n 18 bin/gaspi_cleanup > $GPI2_PATH/bin/gaspi_cleanup
    chmod +x $GPI2_PATH/bin/gaspi_cleanup
else
    cp bin/gaspi_run.ssh $GPI2_PATH/bin/gaspi_run
    cp bin/ssh.spawner $GPI2_PATH/bin/
    cp bin/gaspi_cleanup $GPI2_PATH/bin/
fi
cp bin/gaspi_logger $GPI2_PATH/bin/

cp -r lib64 $GPI2_PATH
cp -r tests $GPI2_PATH
cp -r include $GPI2_PATH


cat << EOF

Installation finished successfully!

Add the following line to your $HOME/.bashrc (or your shell):
PATH=\${PATH}:${GPI2_PATH}/bin

EOF
echo "Success!"  >> install.log

clean_bak_files

exit 0
