#!/bin/sh

GPI2_PATH=/opt/GPI2
OFED_PATH=""
OFED=0
WITH_MPI=0
MPI_PATH=""
WITH_LL=0
WITH_F90=1

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

	     	     
EOF
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
	echo "Error: could not find OFED package."
	echo "Run this script with the -o option and providing the path to your OFED installation."
	echo
	exit 1
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

#build everything
make clean &> /dev/null
make -C src depend &> /dev/null

echo -e -n "\nBuilding GPI..."
make gpi >> install.log 2>&1
if [ $? != 0 ]; then
    echo "Compilation of GPI-2 failed (see install.log)"
    echo "Aborting..."
    exit -1
fi

if [ $WITH_F90 = 1 ]; then
    make fortran >> install.log 2>&1
    if [ $? != 0 ]; then
	echo "Creation of GPI-2 Fortran bindings failed (see install.log)"
	echo "Aborting..."
	exit -1
    fi
fi
echo " done."


echo -e -n "\nBuilding tests..."
make tests >> install.log 2>&1
if [ $? != 0 ]; then
    echo "Compilation of tests failed (see install.log)"
    echo "Aborting..."
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

exit 0
