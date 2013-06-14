#!/bin/sh

GPI2_PATH=/opt/GPI2
OFED_PATH=""
OFED=0
usage()
{
cat << EOF

    Usage: `basename $0` [-p PATH_GPI2_INSTALL] [-o OFED_DIR]
      where
             -p Path where to install GPI-2
             -o Path to OFED installation

EOF
}

while getopts ":p:o:" opt; do
    case $opt in
	o)
	    echo "Path to OFED to be used: $OPTARG" >&2
	    OFED_PATH=$OPTARG
	    OFED=1
	    ;;
        p)
            echo "Path to be used: $OPTARG" >&2
            GPI2_PATH=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
	    usage
	    exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

#check ofed installation
if [ $OFED = 0 ]; then
    echo "Searching OFED installation..."
    INFO=/etc/infiniband/info
    if [ -x $INFO ]; then
	
	OFED_PATH=$(${INFO} | grep -w prefix | cut -d '=' -f 2)
	echo "Found OFED installation in $OFED_PATH"
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

#check ibverbs support (at least RoCE)
grep IBV_LINK_LAYER_ETHERNET $OFED_PATH/include/infiniband/verbs.h > /dev/null
if [ $? != 0 ]; then
    echo "Error: Too old version of libibverbs."
    echo "Please update your OFED stack to a more recent version."
    exit 1
fi
    
sed -i  "s,OFED_PATH = /usr/local/ofed1.51,OFED_PATH = $OFED_PATH,g" src/Makefile
sed -i  "s, OFED_PATH = /usr/local/ofed1.5.4.1,OFED_PATH = $OFED_PATH,g" tests/make.defines

#build everything
make gpi
if [ $? != 0 ]; then
    echo "Compilation of GPI-2failed"
    echo "Aborting..."
    exit -1
fi

make tests
if [ $? != 0 ]; then
    echo "Compilation of tests failed"
    echo "Aborting..."
    exit -1
fi
make docs 2> /dev/null

#copy things to the right place
echo
if [ ! -d "$GPI2_PATH" ]; then
    mkdir -p $GPI2_PATH 2> /dev/null
    if [ "$?" != "0" ] ; then
	echo
        echo "Failed to create directory (${GPI2_PATH}). Check your permissions or choose a different directory."
	echo "You can use the (-p) option to choose a different directory."
	echo "Your installation was not completed!"
	echo
        exit 1
    fi

fi

cp -r bin $GPI2_PATH
cp -r lib64 $GPI2_PATH
cp -r tests $GPI2_PATH
cp -r include $GPI2_PATH

cat << EOF

Installation finished successfully!

Add the following line to your $HOME/.bashrc:
PATH=\${PATH}:${GPI2_PATH}/bin

EOF


exit 0
