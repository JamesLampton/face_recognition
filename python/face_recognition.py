#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to use dlib's face recognition tool.  This tool maps
#   an image of a human face to a 128 dimensional vector space where images of
#   the same person are near to each other and images from different people are
#   far apart.  Therefore, you can perform face recognition by mapping faces to
#   the 128D space and then checking if their Euclidean distance is small
#   enough. 
#
#   When using a distance threshold of 0.6, the dlib model obtains an accuracy
#   of 99.38% on the standard LFW face recognition benchmark, which is
#   comparable to other state-of-the-art methods for face recognition as of
#   February 2017. This accuracy means that, when presented with a pair of face
#   images, the tool will correctly identify if the pair belongs to the same
#   person or is from different people 99.38% of the time.
#
#   Finally, for an in-depth discussion of how dlib's tool works you should
#   refer to the C++ example program dnn_face_recognition_ex.cpp and the
#   attendant documentation referenced therein.
#
#
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  This code will also use CUDA if you have CUDA and cuDNN
#   installed.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
import glob
from skimage import io, transform
from skimage import img_as_ubyte
import numpy as np

if len(sys.argv) != 5:
    print(
        "Call this program like this:\n"
        "   ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces db_out_path\n"
        "You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    exit()

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
faces_folder_path = sys.argv[3]
db_path = sys.argv[4]

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

try:
    win = dlib.image_window()
    do_win = True
except:
    do_win = False

import sqlite3
import json
class FileTracker:
    def __init__(self, db_path):
        print 'Creating conn...'
        self.conn = sqlite3.connect(db_path)
        print 'Creating cursor...'
        self.curs = self.conn.cursor()

        print 'Initializing tables...'
        self.curs.execute('CREATE TABLE IF NOT EXISTS paths (file_id INT PRIMARY KEY, path TEXT, processed INT)')
        self.curs.execute('CREATE TABLE IF NOT EXISTS faces (file_id INT, face_num INT, left INT, top INT, right INT, bottom INT, face_descriptor TEXT)')

        self._paths = {}

        print 'Pulling existings paths...'
        self.curs.execute('SELECT file_id, path, processed FROM paths')
        for file_id, path, processed in self.curs.fetchall():
            self._paths[path] = [file_id, processed]

    def add_path(self, path):
        if path in self._paths: return

        next_id = len(self._paths) + 1
        self._paths[path] = [next_id, 0]
        self.curs.execute('INSERT INTO paths (file_id, path, processed) VALUES (?, ?, ?)', (next_id, path, 0,))

    def add_face(self, file_id, k, left, top, right, bottom, face_descriptor):
        self.curs.execute('INSERT INTO faces (file_id, face_num, left, top, right, bottom, face_descriptor) VALUES (?, ?, ?, ?, ?, ?, ?)', \
                (file_id, k, left, top, right, bottom, json.dumps(np.asarray(face_descriptor).tolist()),))

    def get_unprocessed(self):
        return [(k, v[0],) for (k, v,) in self._paths.items() if v[1] == 0]

    def set_processed(self, path):
        arr = self._paths[path]
        self.curs.execute('UPDATE paths SET processed = 1 WHERE file_id = ?', (arr[0],))
        arr[1] = 1

    def commit(self):
        self.conn.commit()

db_state = FileTracker(db_path)
import fnmatch

print 'Walking', faces_folder_path
for dirpath, dirnames, filenames in os.walk(faces_folder_path):
    for f in filenames:
        if fnmatch.fnmatch(f, '*.[Jj][Pp][Gg]'):
            f = os.path.join(dirpath, f)
            db_state.add_path(f)
            #print f

db_state.commit()
c = 0

# Now process all the images
for f, file_id in db_state.get_unprocessed():
    c += 1
    print("Processing file: {}".format(f))
    try:
        img = io.imread(f)
        if img.shape[0] > 768:
            scale = 768. / img.shape[0]
            #print img.shape, scale
            img = img_as_ubyte(transform.rescale(img, scale))
            #print img.shape
    except:
        print 'Error processing:', f
        continue

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    if do_win and len(dets) > 0:
        win.clear_overlay()
        win.set_image(img)

    # Now process each face we found.
    err = False
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)
        # Draw the face landmarks on the screen so we can see what face is currently being processed.
        if do_win:
            win.clear_overlay()
            win.add_overlay(d)
            win.add_overlay(shape)

        # Compute the 128D vector that describes the face in img identified by
        # shape.  In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people. Here we just print
        # the vector to the screen.
        try:
            face_descriptor = facerec.compute_face_descriptor(img, shape)
        except:
            err = True
            continue
        db_state.add_face(file_id, k, d.left(), d.top(), d.right(), d.bottom(), face_descriptor)

        # It should also be noted that you can also call this function like this:
        #  face_descriptor = facerec.compute_face_descriptor(img, shape, 100)
        # The version of the call without the 100 gets 99.13% accuracy on LFW
        # while the version with 100 gets 99.38%.  However, the 100 makes the
        # call 100x slower to execute, so choose whatever version you like.  To
        # explain a little, the 3rd argument tells the code how many times to
        # jitter/resample the image.  When you set it to 100 it executes the
        # face descriptor extraction 100 times on slightly modified versions of
        # the face and returns the average result.  You could also pick a more
        # middle value, such as 10, which is only 10x slower but still gets an
        # LFW accuracy of 99.3%.

        if do_win:
            dlib.hit_enter_to_continue()

    if not err:
        db_state.set_processed(f)

    if c % 20 == 0:
        print 'Commit...'
        db_state.commit()

    db_state.commit()
