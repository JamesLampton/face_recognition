#!/bin/bash

mkdir dlib/build
cd dlib/build

#sed -i -e 's/OpenCV 2.4.5/OpenCV 3.1.0/' ../CMakeLists.txt
# -find_package(OpenCV 2.4.5 REQUIRED)
# +find_package(OpenCV 3.1.0 REQUIRED)

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4

make install

cd ..
python setup.py install
