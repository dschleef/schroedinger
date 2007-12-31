#!/bin/sh

if [ $(whoami) = wladimir ]; then
  confflags="--prefix=/home/wladimir --with-cuda --disable-gtk-doc"
else
  confflags="--enable-gtk-doc"
fi

autoreconf -i -f &&
  gtkdocize --copy &&
  ./configure --enable-maintainer-mode --disable-static $confflags $@
