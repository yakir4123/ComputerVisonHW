import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2


def bilinear_interpolation(I, x, y):
    x1 = int(np.floor(x))
    x2 = x1 + 1
    y1 = int(np.floor(y))
    y2 = y1 + 1
    interpolation = 0
    if 0 <= x1 < len(I[0]):
        if 0 <= y1 < len(I):
            interpolation += I[y1, x1] * (y2 - y) * (x2 - x)
        if 0 <= y2 < len(I):
            interpolation += I[y2, x1] * (y - y1) * (x2 - x)
    if 0 <= x2 < len(I[0]):
        if 0 <= y1 < len(I):
            interpolation += I[y1, x2] * (y2 - y) * (x - x1)
        if 0 <= y2 < len(I):
            interpolation += I[y2, x2] * (y - y1) * (x - x1)
    return interpolation


def image_warp(I, map_u, map_v):
    I_wrapped = np.zeros(I.shape)
    for x in range(len(I)):
        for y in range(len(I[x])):
            try:
                I_wrapped[y, x] = bilinear_interpolation(I, x - map_u[y, x], y - map_v[y, x])
            except IndexError:
                pass
    return I_wrapped


def computer_exercise2():
    img1 = sio.loadmat('HW5\\hw4_data\\imgs_for_optical_flow.mat')['img1']
    map_u = sio.loadmat('HW5\\hw4_data\\imgs_for_optical_flow.mat')['u']
    map_v = sio.loadmat('HW5\\hw4_data\\imgs_for_optical_flow.mat')['v']
    I_warped = image_warp(img1, map_u, map_v)

    # to convert optical flow to actual remaping
    yy, xx = np.mgrid[0:len(img1), 0:len(img1)]
    # to keep this  dtype float32 and not change it to float64
    map_v = yy - map_v
    map_u = xx - map_u
    std_I_warped = cv2.remap(img1, map_u.astype('float32'), map_v.astype('float32'), interpolation=cv2.INTER_LINEAR)

    show_plots((img1, I_warped, std_I_warped), ('original image', 'warped image', 'cv2.remap()'))
    plt.show()


def remap(tup_index, iT=None):
    return iT.dot(np.array(tup_index.appand(1)))


def affine_wrap(I, iT):
    I_wrapped = np.zeros(I.shape)
    for x in range(len(I)):
        for y in range(len(I[x])):
            try:
                (x_, y_) = iT.dot((x, y) + (1, ))
                I_wrapped[y, x] = bilinear_interpolation(I, x_, y_)
            except IndexError:
                pass
    return I_wrapped


def computer_exercise3():
    img = sio.loadmat('HW5\\hw4_data\\imgs_for_optical_flow.mat')['img1']

    # invertible matrix
    T = np.array([[0.5, 0.2, 0], [0, 0.5, 8]])
    iT = cv2.invertAffineTransform(T)

    warped_img = affine_wrap(img, T)
    std_warped_img = cv2.warpAffine(img, iT, img.shape)
    show_plots((img, warped_img, std_warped_img), ('original image', 'affine warped image', 'standard implementation'))


def show_plots(images, labels):
    fig, axs = plt.subplots(1, len(images))
    for i in range(len(images)):
        axs[i].set(xlabel=labels[i])
        axs[i].imshow(images[i])

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

