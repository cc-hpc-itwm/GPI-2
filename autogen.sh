#!/bin/sh
add=$1

# SUBDIR OPTION FOR MODERN AUTOMAKE
atmk_ver=`automake --version | head -n 1 | awk '{print $NF}'`
if [[ `echo ${atmk_ver%.*} \>= 1.14 | bc` -eq 1 ]]; then
    sed 's/AUTOMAKE_SUBDIRS/subdir-objects/g' configure.ac.in > configure.ac
else
    sed 's/AUTOMAKE_SUBDIRS//g' configure.ac.in > configure.ac
fi

# AUTOTOOLS CONFIGURATION
if [[ ! -e build-aux ]]; then
    mkdir build-aux
fi
autoreconf --install -I build-aux $add
rm -fr autom4te.cache/
