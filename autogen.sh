#!/bin/sh
add=$1

if [[ ! -e build_aux ]]; then
    mkdir build-aux
fi
autoreconf --install -I m4 -I build-aux $add
rm -fr autom4te.cache/
