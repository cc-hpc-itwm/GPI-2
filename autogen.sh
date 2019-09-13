#!/bin/sh
add=$1

if [ ! -e build-aux ]; then
    mkdir build-aux
fi
autoreconf --install -I build-aux $add
rm -fr autom4te.cache/
