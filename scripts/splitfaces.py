#!/usr/bin/python

import sys
import os
import dlib
import glob
import numpy as np
import cv2
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
        "python scripts\splitfaces.py ${image_path} ${output_folder_path}\n")
    exit()

predictor_path = 'models\shape_predictor_68_face_landmarks.dat'
image_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

for f in glob.glob(image_path):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)

    print("Number of faces detected: {}".format(len(dets)))

    dets, scores, idx = detector.run(img, 1)

    index = 0
    for k, d in enumerate(dets):

        print("Detection {}, dets{},score: {}, face_type:{}".format(k, d, scores[k], idx[k]))

        face_type = idx[k]

        rectanglepoints = np.zeros((1, 4), dtype=int)
        rectanglepoints = (d.left(), d.top(), d.right(), d.bottom())        

        x = d.left()
        y = d.top()
        w = d.right() - x
        h = d.bottom() - y

        margin_len = 30
        x = x - margin_len
        y = y - margin_len
        w = w + (margin_len * 2)
        h = h + (margin_len * 2)

        sub_img = img[y:y+h, x:x+w]

        tmps = f.split("\\")
        image_name = tmps[len(tmps) -1]

        if not os.path.exists(faces_folder_path):
            os.makedirs(faces_folder_path)

        new_image_name = faces_folder_path + "\\" + image_name.split('.')[0] + "_" + str(index)+ ".jpg"
        cv2.imwrite(new_image_name , cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR))

        index = index + 1
