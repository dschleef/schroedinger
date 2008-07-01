#!/bin/sh

if [ $(whoami) = wladimir ]; then
  confflags="--prefix=/home/wladimir --with-cuda --disable-gtk-doc"
elif [ $(whoami) = ds ]; then
  #confflags="--disable-gtk-doc --enable-orc"
  confflags="--enable-gtk-doc"
else
  confflags="--enable-gtk-doc"
fi

autoreconf -i -f &&
  ./configure --enable-maintainer-mode --disable-static $confflags $@
