import numpy as np


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
