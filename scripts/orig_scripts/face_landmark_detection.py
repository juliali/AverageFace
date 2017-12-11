#!/usr/bin/python

import sys
import os
import dlib
import glob
import numpy as np
from skimage import io

def shape_to_np(shape):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype = int)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (int(shape.part(i).x), int(shape.part(i).y))
 
    # return the list of (x, y)-coordinates
    return coords

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "python face_landmark_detection.py shape_predictor_68_face_landmarks.dat faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

index = 0
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)

    #print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        # # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        shape_np = shape_to_np(shape)
        np.savetxt(f + '.txt', shape_np, fmt = '%i')
        index = index + 1

print "{} faces have been extracted points.".format(index)
