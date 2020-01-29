import cv2
import numpy as np
import scipy.io as sio
from scipy import sparse
from scipy.sparse import linalg
from matplotlib import pyplot as plt


def computer_exercise4():
    imgs = sio.loadmat('HW4\\hw4_data\\imgs_for_optical_flow.mat')
    I1 = imgs['img1']
    imgs_names = ['img2', 'img3', 'img4', 'img5', 'img6']
    labels = ['I2_t', 'I3_t', 'I4_t', 'I5_t', 'I6_t']
    fig, axs = plt.subplots(1, 5)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(5):
        axs[i].set(xlabel=labels[i])
        axs[i].imshow(imgs[imgs_names[i]] - I1, cmap='gray')
    plt.show()


def horn_n_schunck_sparse_matrix(I, lambda_):
    h_row, h_col = cv2.getDerivKernels(1, 0, 3)
    hx = h_col.dot(h_row.T)
    hy = hx.T

    Ix = cv2.filter2D(I, -1, hx, borderType=cv2.BORDER_CONSTANT)
    Iy = cv2.filter2D(I, -1, hy, borderType=cv2.BORDER_CONSTANT)

    N_cols = np.size(I, 1)
    N = np.size(I, 0) * N_cols
    # offset for every line
    u_indices = np.array([-2 * N_cols, -2, 0, 1, 2, 2 * N_cols])
    v_indices = np.array([-2 * N_cols + 1, -1, 0, 1, 3, 2 * N_cols + 1])

    # i_s on the paper run from 1 to N
    # calculate u and v separate
    u_rows = np.array([u_indices + 2 * i_s for i_s in range(0, N)])
    v_rows = np.array([v_indices + 2 * i_s for i_s in range(0, N)])

    # concat it for 1 long row for the sparse matrix
    rows = np.empty((2 * N, 6), dtype='int64')
    rows[0::2] = u_rows
    rows[1::2] = v_rows
    # make the indices start from 0 and not from 1
    rows = rows.ravel()

    # every col has exactly len(indices) numbers
    cols = np.array([[i] * len(u_indices) for i in range(2 * N)], dtype='int64').ravel()

    # inf means empty place that will be deleted in the future to handle borders
    data = float('inf') * np.ones(6 * 2 * N)

    # for u' rows
    for i_s in range(0, N):
        i = i_s // N_cols
        j = i_s % N_cols
        # border indices are out of bound need to remove them
        inside_indices = np.where((0 <= u_rows[i_s]) & (u_rows[i_s] < 2 * N))[0]
        # calculate the data of this row
        row_data = np.array([-2 * lambda_, -2 * lambda_,
                             Ix[i, j] * Ix[i, j] + 2 * (len(inside_indices) - 2) * lambda_,
                             Ix[i, j] * Iy[i, j], -2 * lambda_, -2 * lambda_])
        # place it on its place, its multiply 2 to leave place for v' rows
        data[inside_indices + 6 * (2 * i_s)] = row_data[inside_indices]

    # for v' rows
    for i_s in range(0, N):
        i = i_s // N_cols
        j = i_s % N_cols
        # border indices are out of bound
        inside_indices = np.where((0 <= v_rows[i_s]) & (v_rows[i_s] < 2 * N))[0]
        # calculate the data of this row
        row_data = np.array([-2 * lambda_, -2 * lambda_, Ix[i, j] * Iy[i, j],
                             Iy[i, j] * Iy[i, j] + 2 * (len(inside_indices) - 2) * lambda_,
                             -2 * lambda_, -2 * lambda_])
        # place it on its place
        data[inside_indices + 6 * (2 * i_s + 1)] = row_data[inside_indices]

    # get rid of out of bound indices
    in_bound = np.where(data != float('inf'))
    data = data[in_bound]
    rows = rows[in_bound]
    cols = cols[in_bound]
    return sparse.coo_matrix((data, (rows, cols)))


def horn_n_schunck_optical_flow(I_1, I_2, lambda_):
    A = horn_n_schunck_sparse_matrix(I_1, lambda_)

    # calculate b
    h_row, h_col = cv2.getDerivKernels(1, 0, 3)
    hx = h_col.dot(h_row.T)
    hy = hx.T

    Ix = cv2.filter2D(I_1, -1, hx, borderType=cv2.BORDER_CONSTANT).ravel()
    Iy = cv2.filter2D(I_1, -1, hy, borderType=cv2.BORDER_CONSTANT).ravel()
    It = (I_2 - I_1).ravel()
    N = np.size(I_1, 0) * np.size(I_1, 1)
    b = np.zeros(2 * N)
    b[::2] = Ix*-It
    b[1::2] = Iy*-It
    theta = linalg.lsqr(A, b)
    return theta[0][::2], theta[0][1::2]


def computer_exercise5(lambda_):
    imgs = sio.loadmat('HW4\\hw4_data\\imgs_for_optical_flow.mat')
    I1 = imgs['img1']
    imgs_names = ['img2', 'img3', 'img4', 'img5', 'img6']

    fig, axs = plt.subplots(len(imgs_names), 4, figsize=(26, 20))
    fig.suptitle('Lambda = {}'.format(lambda_), fontsize=30)

    # Every row plt images
    for i in range(len(imgs_names)):
        # image 1
        axs[i, 0].imshow(I1)
        axs[i, 0].set_ylabel(imgs_names[i], fontsize=40)
        axs[i, 0].set_xlabel('img1', fontsize=30)

        # t + 1  image
        I = imgs[imgs_names[i]]
        axs[i, 1].imshow(I)
        axs[i, 1].set_xlabel(imgs_names[i], fontsize=30)

        # optical flow image
        X, Y = np.meshgrid(range(I.shape[0]), range(I.shape[1]))
        U, V = horn_n_schunck_optical_flow(I1, I, lambda_)
        axs[i, 2].quiver(X, Y, U, V)
        axs[i, 2].set_xlabel('optical flow', fontsize=30)

        # to convert optical flow to actual remaping
        yy, xx = np.mgrid[0:len(I), 0:len(I)]

        # wrap image back to img 1
        U = xx - U.reshape(I.shape)
        V = yy - V.reshape(I.shape).astype('float32')

        U = U.astype('float32')
        V = V.astype('float32')
        wraped = cv2.remap(I, U, V, cv2.INTER_LINEAR)
        axs[i, 3].imshow(wraped)
        axs[i, 3].set_xlabel('warped image', fontsize=30)
        mse = (np.square(I1 - wraped)).mean(axis=None)
        axs[i, 3].yaxis.set_label_position("right")
        axs[i, 3].set_ylabel('MSE = {0:.3f}'.format(mse), fontsize=30)

    for ax in axs.flat:
        ax.set(aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
