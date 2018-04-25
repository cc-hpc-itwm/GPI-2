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
GPI2_DEVICE=IB
usage()
{
cat << EOF

GPI2 Installation:

    Usage: `basename $0` [-p PATH_GPI2_INSTALL] <further options>
      where
             -p Path where to install GPI-2 (default: ${GPI2_PATH})

    Further options:
             --with-mpi<=path>              Use this option if you aim at mixed mode with MPI.
	                                    See README for more information.

	     --with-ll                      Use this option if you have Load Leveler as batch system.
	                                    This integrates with Load Leveler and uses poe as application launcher.

	     --with-ethernet                Build GPI-2 for Ethernet (only if you don't have Infiniband).
	                                    See README for more information.

             --with-infiniband<=path>       Build GPI-2 for Infiniband hardware (this is the default).
                                            You can provide the path to your OFED installation.

             --with-fortran=(true,false)    Enable/Disable Fortran bindings (default: enabled).

	     --with-cuda<=path>             Use this option if you aim mixed use with CUDA and GPU support.
	     	     
EOF
}

check_util_exists()
{
    which $1 > /dev/null 2>&1
    if [ $? != 0 ]; then
	printf '\nThis version of GPI-2 requires the %s utility.\n\n' "${1}"
	exit 1
    fi
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

    if [ -r src/Makefile.bak ]; then
	mv src/Makefile.bak src/Makefile
    fi
}

check_compilers()
{
  CC=${CC:-gcc}
  CXX=${CXX:-g++}
  FC=${FC:-gfortran}
}

#check some requirements
check_util_exists gawk
check_util_exists sed
check_compilers

while getopts ":hp:-:" opt; do
    case $opt in
	-)
	    case "${OPTARG}" in
		with-mpi)
		    which mpirun > /dev/null 2>&1
		    if [ $? != 0 ]; then
			echo "Couldn't find MPI installation. Please provide path to your MPI installation."
			echo "    ./install.sh <other options> --with-mpi=<Path to MPI installation>"
			echo ""
			exit 1
		    fi
		    MPI_BIN=`which mpirun`
		    MPI_PATH=`dirname $MPI_BIN`
		    MPI_PATH=`dirname $MPI_PATH`
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
		with-ethernet)
		    echo "With Ethernet support" >&2;
		    GPI2_DEVICE=TCP
		    ;;
		with-infiniband)
		    echo "With Infiniband support" >&2;
		    GPI2_DEVICE=IB
		    ;;
		with-infiniband=*)
		val=${OPTARG#*=}
		if [ "$val" = "" ]; then
		    echo "Forgot to provide OFED path?"
		    exit 1
		fi
		OFED_PATH=$val
		OFED=1
		opt=${OPTARG%=$val}
		echo "With Infiniband support (${OFED_PATH})" >&2;
		GPI2_DEVICE=IB
		;;
		with-fortran)
		    WITH_F90=1
		    ;;
		with-fortran=*)
		val=${OPTARG#*=}
		if [ "$val" = "false" ]; then
		    WITH_F90=0
		    sed -i "s,fortran,,g" tests/tests/Makefile
		fi
		;;
		with-cuda)
                    which nvcc > /dev/null 2>&1
                    if [ $? != 0 ]; then
                        echo "Couldn't find CUDA installation. Please provide path to your CUDA installation."
			echo "    ./install.sh <other options> --with-cude=<Path to CUDA installation>"
			echo ""
                        exit 1
                    fi
		    NVCC_BIN=`which nvcc`
		    CUDA_PATH=`dirname $NVCC_BIN`
		    CUDA_PATH=`dirname $CUDA_PATH`
		    
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
        p)
            echo "Installation path to be used: $OPTARG" >&2
            GPI2_PATH=$OPTARG
            ;;
        h)
	    usage
	    exit 1
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

#save defaults
cp src/make.inc src/make.inc.bak
cp tests/make.defines tests/make.defines.bak

echo "$0 $@" >> install.log

if [ $GPI2_DEVICE = IB ]; then
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
		echo "Run this script with the --with-infiniband=<path> option and provide the path to your OFED installation."
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

    if [ -n "$GPI2_EXTRA_CFLAGS" ]; then
	echo "###### added by install script" >> src/make.inc
	echo "CFLAGS += ${GPI2_EXTRA_CFLAGS}" >> src/make.inc
	echo "DBG_CFLAGS += ${GPI2_EXTRA_CFLAGS}" >> src/make.inc
	echo "export" >> src/make.inc
    fi
else
    sed -i  "s,-libverbs,,g" tests/make.defines
    echo "###### added by install script" >> src/make.inc
    echo "GPI2_DEVICE = TCP" >> src/make.inc
    sed -i "s,GASPI_IB,GASPI_ETHERNET,g" tests/defs/*.def
    sed -i "s,GASPI_IB,GASPI_ETHERNET,g" tests/tests/test_utils.h
fi

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
	    echo "    ./install.sh <other options> --with-mpi=<Path to MPI installation>"
	    echo ""
	    exit 1
	fi
    fi

    if [ -r $MPI_PATH/lib64/libmpi.so ] || [ -r $MPI_PATH/lib64/libmpi.a ]; then
	MPI_LIB_PATH=$MPI_PATH/lib64
	MPI_LIB=mpi
    else
	if [ -r $MPI_PATH/lib/libmpi.so ] || [ -r $MPI_PATH/lib/libmpi.a ]; then
	    MPI_LIB_PATH=$MPI_PATH/lib
	    	MPI_LIB=mpi
	else
	    if [ -r $MPI_PATH/lib64/libmpich.so ] || [ -r $MPI_PATH/lib64/libmpich.a ]; then
		MPI_LIB_PATH=$MPI_PATH/lib64
		MPI_LIB=mpich
	    else
		if [ -r $MPI_PATH/lib/libmpich.so ] || [ -r $MPI_PATH/lib/libmpich.a ]; then
		    MPI_LIB_PATH=$MPI_PATH/lib
		    	MPI_LIB=mpich
		else
		    echo "Cannot find libmpi (or libmpich). Please provide path to MPI installation."
		    echo "    ./install.sh <other options> --with-mpi=<Path to MPI installation>"
		    echo ""
		    exit 1
		fi
	    fi
	fi
    fi
    
    echo "Using MPI: ${MPI_PATH}" | tee -a install.log
    echo "###### added by install script" >> src/make.inc
    echo "CFLAGS += -DGPI2_WITH_MPI" >> src/make.inc
    echo "DBG_CFLAGS += -DGPI2_WITH_MPI" >> src/make.inc
    echo "INCLUDES += -I${MPI_INC_PATH}" >> src/make.inc

    echo "###### added by install script" >> tests/make.defines
    echo "CFLAGS += -DGPI2_WITH_MPI -I${MPI_INC_PATH} ${GPI2_EXTRA_CFLAGS}" >> tests/make.defines
    echo "LIB_PATH += -L${MPI_LIB_PATH} ${GPI2_EXTRA_LIBS_PATH}" >> tests/make.defines
    echo "LIBS += -l${MPI_LIB} ${GPI2_EXTRA_LIBS}" >> tests/make.defines
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
    #check device: for now only IB is working
    if [ $GPI2_DEVICE = TCP ]; then
	echo "GPU (Cuda) support is only available with Infiniband."
	echo ""
	exit 1
    fi
    #check module
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
make clean > /dev/null 2>&1
which makedepend > /dev/null 2>&1
if [ $? = 0 ]; then
    make -C src depend > /dev/null 2>&1
fi

NCORES=`grep -c '^processor' /proc/cpuinfo`
if [ -z $NCORES ]; then
    NCORES=1
fi

printf "\nBuilding GPI..."
make -j$NCORES gpi >> install.log 2>&1
if [ $? != 0 ]; then
    echo "Compilation of GPI-2 failed (see install.log)"
    echo "Aborting..."
    clean_bak_files
    exit 1
fi

if [ $WITH_F90 = 1 ]; then
    make fortran >> install.log 2>&1
    if [ $? != 0 ]; then
	echo "Creation of GPI-2 Fortran bindings failed (see install.log)"
	echo "Aborting..."
	clean_bak_files
	exit 1
    fi
fi
echo " done."


printf "\nBuilding tests..."
make -j$NCORES V=3 tests >> install.log 2>&1
if [ $? != 0 ]; then
    echo "Compilation of tests failed (see install.log)"
    echo "Aborting..."
    clean_bak_files
    exit 1
fi
echo " done."

printf "\nCreating documentation..."
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

if [ $GPI2_DEVICE = TCP ]; then
    sed -i "s,GASPI_ETHERNET,GASPI_IB,g" tests/defs/*.def
    sed -i "s,GASPI_ETHERNET,GASPI_IB,g" tests/tests/test_utils.h
fi

cat <<PKG > GPI2.pc
prefix=${GPI2_PATH}
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib64
includedir=\${prefix}/include

Name: GPI-2
Description: GPI-2 library
Version: 1.3.0
Cflags: -I\${includedir}
Libs: -L\${libdir} -lGPI2
PKG
if [ $GPI2_DEVICE = IB ]; then
    echo "Libs.private: ${OFED_PATH}/lib64 -libverbs -lpthread" >> GPI2.pc
else
    echo "Libs.private: -lrt -lpthread" >> GPI2.pc
fi

mkdir $GPI2_PATH/lib64/pkgconfig/
mv GPI2.pc $GPI2_PATH/lib64/pkgconfig/

exit 0
