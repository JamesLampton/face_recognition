#!/bin/bash

cd boost_1_65_1

./bootstrap.sh --with-libraries=python --prefix=/usr
./b2
./b2 install

ldconfig
