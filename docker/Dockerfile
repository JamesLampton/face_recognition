# Base image for piggybank-squeal stuff
FROM fedora:26
MAINTAINER James Lampton <jlampton@gmail.com>

RUN dnf update -y

ADD dlib_face_recognition_resnet_model_v1.dat.bz2 /
ADD shape_predictor_68_face_landmarks.dat.bz2 /

ADD dlib-v19.5.tar.gz /
ADD boost_1_65_1.tar.gz /

RUN dnf install -y cmake gcc gcc-c++ python make python-devel python2-scikit-image libX11-devel openblas-devel lapack-devel which python2-matplotlib bzip2
# qt5-qtbase python-qt5 python2-matplotlib-qt5

ADD build_boost.sh /
ADD build_dlib.sh /

RUN /build_boost.sh && /build_dlib.sh

RUN bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2 && bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# Setup google-drive-ocamlfuse
ADD opam-full-1.2.2.tar.gz /
  
RUN dnf install -y ocaml.x86_64 patch m4 unzip fuse-devel gmp-devel libcurl-devel perl-Pod-Html sqlite-devel zlib-devel redhat-rpm-config ocaml-ocamldoc python2-scikit-learn

ADD build_opam.sh /
RUN /build_opam.sh

# eval `opam config env`

# docker run --rm=true -v /home/jhl1/git/face_recognition/linky/:/linky -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -ti jhlampt/dlibn/bash

# docker run --rm=true -v /home/jhl1/git/face_recognition/linky/:/linky -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/jhl1/Documents/My\ Pictures/:/Pictures -ti jhlampt/dlib /bin/bash

# --cap-add SYS_ADMIN --privileged
