#!/bin/sh
add=$1

# SUBDIR OPTION FOR MODERN AUTOMAKE
## Automake versions >=1.14 output the warning message:
## option is disabled:
##   "warning: possible forward-incompatibility.
##    At least a source file is in a subdirectory, but the 'subdir-objects'
##    automake option hasn't been enabled.  For now, the corresponding output
##    object file(s) will be placed in the top-level directory.  However,
##    this behaviour will change in future Automake versions: they will
##    unconditionally cause object files to be placed in the same subdirectory
##    of the corresponding sources.
##    You are advised to start using 'subdir-objects' option throughout your
##    project, to avoid future incompatibilities"

atmk_ver=`automake --version | head -n 1 | awk '{print $NF}'`
atmk_mjver=`echo $atmk_ver | awk -F "." {'print $1'}`
atmk_miver=`echo $atmk_ver | awk -F "." {'print $2'}`

if [ "$atmk_mjver" -ge 1 -a "$atmk_miver" -ge 14 ]; then
    sed 's/AUTOMAKE_SUBDIRS/subdir-objects/g' configure.ac.in > configure.ac
else
    sed 's/AUTOMAKE_SUBDIRS//g' configure.ac.in > configure.ac
fi

# AUTOTOOLS CONFIGURATION
if [ ! -e build-aux ]; then
    mkdir build-aux
fi
autoreconf --install -I build-aux $add
rm -fr autom4te.cache/
