import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from time import time

from api import PRN
from utils.write import write_obj
from utils.estimate_pose import estimate_pose
from utils.cv_plot import *

import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
prn = PRN(is_dlib = True, is_opencv = True)
def test(img):
# read image
    #image = cv2.imread(img)

    # the core: regress position map
    pos = prn.process(img) # use dlib to detect face

    # -- Basic Applications
    # get landmarks
    kpt = prn.get_landmarks(pos)
    # 3D vertices
    vertices = prn.get_vertices(pos)
    # corresponding colors
    colors = prn.get_colors(img, vertices)

    # -- More
    # estimate pose
    camera_matrix, pose = estimate_pose(vertices)
    return img, camera_matrix, kpt,vertices

    # ---------- Plot
    #print (vertices.shape)
    #image_pose = plot_pose_box(image, camera_matrix, kpt)
    #cv2.imshow('sparse alignment', plot_kpt(image, kpt))
    #cv2.imshow('dense alignment', plot_vertices(image, vertices))
    #cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))
    #cv2.waitKey(0)
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    image, camera_matrix, kpt,vertices=test(frame)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    #cv2.imshow('frame',frame)
    cv2.imshow('sparse alignment', plot_kpt(image, kpt))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
