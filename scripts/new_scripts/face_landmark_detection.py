#!/usr/bin/python

import sys
sys.path.insert(0, 'C:\Projects\caffe\python')
import caffe

import sys
import os
import dlib
import glob
import numpy as np
from skimage import io
from shutil import copyfile


def shape_to_np(shape):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype = int)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (int(shape.part(i).x), int(shape.part(i).y))
 
    # return the list of (x, y)-coordinates
    return coords

if len(sys.argv) < 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "python face_landmark_detection.py shape_predictor_68_face_landmarks.dat faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

mean_filename='new_models\mean_image.binaryproto'
gender_net_model_file = 'new_models\deploy_gender.prototxt'
gender_net_pretrained = 'new_models\caffenet_train_iter_10000.caffemodel'

#mean_filename='models\mean.binaryproto'
#gender_net_model_file = 'models\deploy_gender.prototxt'
#gender_net_pretrained = 'models\gender_net.caffemodel'

proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]


gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                              mean=mean,
                              channel_swap=(2, 1, 0),
                              raw_scale=255,
                              image_dims=(256, 256))

gender_list = ['Male', 'Female']

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

with_gender = False

if len(sys.argv) == 4:
    if (sys.argv[3] == 'gender'):
        with_gender = True

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

        tmps = f.split("\\")
        filename = tmps[len(tmps) - 1]
        dirname = f.replace(filename, "")

        if with_gender == True:
            cropped_face = img[d.top():d.bottom(), d.left():d.right(), :]

            h = d.bottom() - d.top()
            w = d.right() - d.left()
            hF = int(h * 0.1)
            wF = int(w*0.1)

            cropped_face_big = img[d.top() - hF:d.bottom() + hF, d.left() - wF:d.right() + wF, :]

            prediction = gender_net.predict([cropped_face_big])
            gender = gender_list[prediction[0].argmax()].lower()
            print 'predicted gender:', gender

            dirname = dirname + gender + "\\"

            if not os.path.exists(dirname):
                os.makedirs(dirname)

            copyfile(f, dirname + filename)

        np.savetxt( dirname + filename + '.txt', shape_np, fmt = '%i')
        index = index + 1

print "{} faces have been extracted points.".format(index)
