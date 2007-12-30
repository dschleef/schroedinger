#!/bin/sh

if [ $(whoami) = wladimir ]; then
  export CFLAGS="-O3 -g"
  autoreconf -i -f &&
  ./configure --prefix=/home/wladimir --with-cuda=/usr/local/cuda \
    --disable-gtk-doc $@
else
  autoreconf -i -f &&
  ./configure --enable-maintainer-mode --disable-static --enable-gtk-doc $@
fi

