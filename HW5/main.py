import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2

import HW5.imageWrapping as imageWrapping


def computer_exercise1():
    img1 = sio.loadmat('HW5\\hw4_data\\imgs_for_optical_flow.mat')['img1']
    map_u = sio.loadmat('HW5\\hw4_data\\imgs_for_optical_flow.mat')['u']
    map_v = sio.loadmat('HW5\\hw4_data\\imgs_for_optical_flow.mat')['v']
    I_wrapped = imageWrapping.image_warp(img1, map_u, map_v)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img1)
    axs[1].imshow(I_wrapped)

    # to convert optical flow to actual remaping
    yy, xx = np.mgrid[0:len(img1), 0:len(img1)]
    # to keep this  dtype float32 and not change it to float64
    map_v = yy - map_v
    map_u = xx - map_u
    axs[2].imshow(cv2.remap(img1, map_u.astype('float32'), map_v.astype('float32'), interpolation=cv2.INTER_LINEAR))
    plt.show()


def main():
    computer_exercise1()

