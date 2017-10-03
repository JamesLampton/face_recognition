#!/usr/bin/env python

import sqlite3
import json
import numpy as np
import sys
import subprocess
import os
import math
import random

from sklearn.cluster import DBSCAN
from skimage import io, transform
from skimage import img_as_ubyte
from sklearn.neighbors import NearestNeighbors

def load_faces(db_file):
    conn = sqlite3.connect(db_file)
    curs = conn.cursor()

    # Load the faces.
    face_vecs = []
    face_ids = []
    curs.execute('SELECT file_id, face_num, face_descriptor FROM faces;')

    for file_id, face_num, face_descriptor in curs.fetchall():
        face_ids.append((file_id, face_num,))
        face_vecs.append(json.loads(face_descriptor))

    face_vecs = np.array(face_vecs)

    return face_ids, face_vecs

def cluster_faces(face_vecs, eps=.3):
    dbs = DBSCAN(eps=eps, n_jobs=-1)

    dbs.fit(face_vecs)

    labels = dbs.labels_

    print 'Found %s labels...' % (len(set(labels)) - (1 if -1 in labels else 0))

    label_counts = {}
    for i in dbs.labels_: 
        label_counts[i] = label_counts.get(i,0) + 1

    return dbs, label_counts

def precompute_distances(face_vecs):
    neigh = NearestNeighbors(radius=1.5)

    neigh.fit(face_vecs)

    A = neigh.radius_neighbors_graph(face_vecs, mode='distance')

    return A.toarray().reshape((-1, 1))

def extract_averages(face_vecs, labels, label_counts):
    avgs = {}
    for cur, label in zip(face_vecs, labels):
        if label == -1: continue

        if label not in avgs:
            avgs[label] = np.zeros(face_vecs.shape)

        avgs[label] += cur

    for label, c in label_counts.items():
        avgs[label] /= c

    return avgs

def open_and_scale(f):
    img = io.imread(f)
    if img.shape[0] > 768:
        scale = 768. / img.shape[0]
        #print img.shape, scale
        img = img_as_ubyte(transform.rescale(img, scale))
    return img

def display_labels(db_file, face_ids, labels, label_counts, out_dir):
    conn = sqlite3.connect(db_file)
    curs = conn.cursor()

    max_faces = 25

    shuffled = range(len(labels))
    random.shuffle(shuffled)

    for label, lab_count in sorted(label_counts.items()):
        #if label == -1: continue
        # FIXME: For now, skip 0
        #if label == 0: continue

        # Create an image array to store the faces
        gs = int(math.ceil(math.sqrt(min(lab_count, max_faces))))
        print 'Label %s [%s] - combined will be %s by %s grid.' % (label, lab_count, gs, gs)

        combined = np.zeros((96*gs, 96*gs, 3), np.float64)

        # Pull some face_ids
        c = 0
        for i in shuffled:
            if labels[i] != label: continue
            if c >= max_faces: break

            #print '%s -- %s / %s' % (label, c, lab_count,)
            face_id = face_ids[i]

            # Pull the path.
            curs.execute('SELECT path FROM paths WHERE file_id = ?', (face_id[0],))
            path = curs.fetchone()[0]
            #print label, c, path

            # Load the image
            img = open_and_scale(path)

            # Pull the mask parameters
            curs.execute('SELECT left, top, right, bottom FROM faces WHERE file_id = ? AND face_num = ?', face_id)
            left, top, right, bottom = curs.fetchone()

            #print i, left, top, right, bottom, img.shape
            try:
                img = img[top:bottom, left:right, :]
                img = transform.resize(img, (96, 96, img.shape[2],))
                x = 96*(c / gs)
                y = 96*(c % gs)
                #print x, y
                combined[ x:x+96, y:y+96, : ] = img
            except:
                continue

            #img[:, left, :] = 0
            #img[:, right, :] = 0
            #img[top, :, :] = 0
            #img[bottom, :, :] = 0


            #io.imsave('tmp.jpg', img)
            #subprocess.call(["display", "tmp.jpg"])
            #os.unlink('tmp.jpg')

            c += 1

        io.imsave('%s/face-%s.jpg' % (out_dir, label), combined)
        #break

def save_clusters(db_file, face_ids, labels):
    conn = sqlite3.connect(db_file)
    curs = conn.cursor()

    curs.execute('CREATE TABLE IF NOT EXISTS cluster_labels (file_id INT, face_num INT, label INT);')

    for label, (file_id, face_num) in zip(labels, face_ids):
        curs.execute('INSERT INTO cluster_labels (file_id, face_num, label) VALUES (?, ?, ?)', (file_id, face_num, label,))

    conn.commit()

def do_main():
    db_file = sys.argv[1]
    out_dir = sys.argv[2]

    print 'Loading faces...'
    face_ids, face_vecs = load_faces(db_file)

    print 'Clustering faces...'
    dbs, label_counts = cluster_faces(face_vecs, eps=.35)
    labels = dbs.labels_

    print 'Saving clusters...'
    save_clusters(db_file, face_ids, labels)

    print 'Saving exemplars...'
    display_labels(db_file, face_ids, labels, label_counts, out_dir)

if __name__ == '__main__': do_main()
