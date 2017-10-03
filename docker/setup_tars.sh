#!/bin/bash

if [ ! -f dlib-v19.5.tar.gz ]
then
    git clone https://github.com/davisking/dlib.git

    cd dlib
    git checkout v19.5
    cd ..

    tar czvf dlib-v19.5.tar.gz --exclude .git dlib/

    rm -fr dlib
fi

test -f boost_1_65_1.tar.gz || wget https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz

test -f shape_predictor_68_face_landmarks.dat.bz2 || wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
test -f dlib_face_recognition_resnet_model_v1.dat.bz2 || wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

test -f opam-full-1.2.2.tar.gz || wget https://github.com/ocaml/opam/releases/download/1.2.2/opam-full-1.2.2.tar.gz

