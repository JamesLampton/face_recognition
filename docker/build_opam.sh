#!/bin/bash

cd /opam-full-1.2.2

./configure
make lib-ext
make
make install

opam init -y
eval `opam config env`
opam update
opam install depext
#opam depext google-drive-ocamlfuse
opam install google-drive-ocamlfuse -y
