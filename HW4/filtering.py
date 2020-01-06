import cv2
import numpy as np
import scipy.misc
import scipy.io as sio
import scipy.linalg as linalg
from matplotlib import pyplot as plt


def computer_exercise1():
    images = 4
    fig, axs = plt.subplots(1, images)

    cols = ['original image', '7x7 Gaussian blur', '21x21 Gaussian blur', '21x21 mean filter']

    for i in range(0, images):
        axs[i].set(xlabel=cols[i])
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # original image
    mandrill = cv2.cvtColor(cv2.imread('HW4\\hw4_data\\mandrill.png'), cv2.COLOR_BGR2RGB)
    axs[0].imshow(mandrill)

    gau7 = cv2.GaussianBlur(src=mandrill, ksize=(7, 7), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_CONSTANT)
    axs[1].imshow(gau7)

    gau21 = cv2.GaussianBlur(src=mandrill, ksize=(21, 21), sigmaX=10, sigmaY=10, borderType=cv2.BORDER_CONSTANT)
    axs[2].imshow(gau21)

    mean21 = cv2.blur(mandrill, (21, 21), borderType=cv2.BORDER_CONSTANT)
    axs[3].imshow(mean21)

    plt.show()


def kernel_as_big_matrix_operation(kernel, M, N):
    dim = M*N
    kernel_dim = len(kernel[0])
    H = np.zeros((dim, dim + 2*(kernel_dim//2 + N)))
    for i in range(dim):
        for kernel_row in range(-(kernel_dim//2), kernel_dim//2 + 1):
            j_start = (i + kernel_row * N) + N
            j_end = (i + kernel_row * N) + kernel_dim + N
            H[i, j_start:j_end] = kernel[kernel_row + kernel_dim//2]
    return H[:, kernel_dim//2 + N:-(kernel_dim//2 + N)]


def computer_exercise2(M, N):
    h1 = np.ones((3, 3)) / 9
    h2 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]) / 8
    h3 = h2.T
    kernels = (h1, h2, h3)
    Hs = [kernel_as_big_matrix_operation(kernel, M, N) for kernel in kernels]

    fig, axs = plt.subplots(1, len(kernels))

    cols = ['3x3 mean filter', 'sobel filter', 'transformed sobel filter', '21x21 mean filter']

    for i in range(0, len(kernels)):
        axs[i].set(xlabel=cols[i])
        pos = axs[i].imshow(Hs[i], interpolation=None)
        fig.colorbar(pos, ax=axs[i], fraction=0.046, pad=0.04)

    for ax in axs.flat:
        ax.set_xticks([])
        # ax.set_yticks([])
    plt.show()


def problem4():
    sample_image = np.array([i + j for i in range(10) for j in range(10)]).astype('float64')
    sample_image = sample_image.reshape((10, 10))
    kernel = np.zeros((5, 5))
    kernel[3, 4] = 1
    shifted_image = cv2.filter2D(sample_image, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    fig, axs = plt.subplots(1, 2)
    pos = axs[0].imshow(sample_image, cmap='gray')
    fig.colorbar(pos, ax=axs[0], fraction=0.046, pad=0.04)
    axs[1].imshow(shifted_image, cmap='gray')
    cols = ['original image', 'shifted image']
    for i in range(0, 2):
        axs[i].set(xlabel=cols[i])
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def computer_exercise3():
    mdict = sio.loadmat('HW4\\hw4_data\\are_these_separable_filters.mat')
    for k in mdict.keys():
        if not k.startswith("__"):
            U, S, V = linalg.svd(mdict[k])
            if list(np.isclose(np.zeros(S.shape), S, atol=1e-12)).count(False) == 1:
                print(k, "S = ", S)


def computer_exercise4(sigColor, sigSpace):
    img_noisy = sio.loadmat('HW4\\hw4_data\\bilateral.mat')['img_noisy']
    bilateral_filtered = cv2.bilateralFilter(img_noisy, -1, sigColor, sigSpace, borderType=cv2.BORDER_CONSTANT)
    fig, axs = plt.subplots(2, 2)
    pos = axs[0, 0].imshow(img_noisy, cmap='gray')
    fig.colorbar(pos, ax=axs[0, 0], fraction=0.046, pad=0.04)
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    pos = axs[0, 1].imshow(bilateral_filtered, cmap='gray')
    fig.colorbar(pos, ax=axs[0, 1], fraction=0.046, pad=0.04)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[1, 0].plot(range(len(img_noisy[len(img_noisy)//2])), img_noisy[len(img_noisy)//2])
    axs[1, 1].plot(range(len(bilateral_filtered[len(bilateral_filtered)//2])),
                   bilateral_filtered[len(bilateral_filtered)//2])
    cols = ['noisy image', 'bilateral filtered']
    for i in range(0, 2):
        axs[1, i].set(xlabel=cols[i])
    plt.show()


def partial_x(filter, sigma):
    res = filter.copy()
    offset = len(filter) // 2
    for y in range(len(filter)):
        for x in range(len(filter[y])):
            # in python rows its the first argument
            res[y, x] *= (x - offset)/sigma
    return res


def computer_exercise5(kernel_size, sigma, theta_start, theta_end, count):
    # ascent = cv2.cvtColor(cv2.imread('HW4\\hw4_data\\ascent.jpg'), cv2.COLOR_BGR2GRAY)
    ascent = scipy.misc.ascent()
    fig, axs = plt.subplots(2, count)
    fig.suptitle('kernel size = {} sigma = {}'.format(kernel_size, sigma), fontsize=16)

    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    filter = np.outer(kernel, kernel.transpose())
    filter_x = partial_x(filter, sigma)
    filter_y = partial_x(filter.T, sigma).T

    theta_delta = (theta_end - theta_start) / count
    theta = theta_start
    for i in range(count):
        directional_filter = filter_x * np.cos(theta) + filter_y * np.sin(theta)
        pos = axs[0, i].imshow(directional_filter, cmap='jet')
        fig.colorbar(pos, ax=axs[0, i], fraction=0.046, pad=0.04)
        axs[0, i].set_title('theta={:.3f}'.format(theta))
        ascent_filtered = cv2.filter2D(ascent, -1, directional_filter, borderType=cv2.BORDER_CONSTANT)
        axs[1, i].imshow(ascent_filtered, cmap='gray')

        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

        theta += theta_delta
    plt.subplots_adjust( wspace=0.1, hspace=-1)
    # plt.subplots_adjust(top=0.85, bottom=0.75, left=0.2, right=0.85, wspace=-0.1, hspace=1)
    plt.show()


def computer_exercise6():
    I = sio.loadmat('HW4\\hw4_data\\imgs_for_optical_flow.mat')['img1']
    h_row, h_col = cv2.getDerivKernels(1, 0, 3)
    hx = h_col.dot(h_row.T)
    hy = hx.T
    fig, axs = plt.subplots(1, 5)
    cols = ['I', 'dI/dx', 'dI/dy', 'dI/dxx', 'dI/dyy']

    # original image
    axs[0].imshow(I, cmap='gray')

    # dI/dx
    Ix = cv2.filter2D(I, -1, hx, borderType=cv2.BORDER_CONSTANT)
    axs[1].imshow(Ix, cmap='gray')

    # dI/dy
    Iy = cv2.filter2D(I, -1, hy, borderType=cv2.BORDER_CONSTANT)
    axs[2].imshow(Iy, cmap='gray')

    # dI/dxx
    Ixx = cv2.filter2D(Ix, -1, hx, borderType=cv2.BORDER_CONSTANT)
    axs[3].imshow(Ixx, cmap='gray')

    # dI/dy
    Iyy = cv2.filter2D(Iy, -1, hy, borderType=cv2.BORDER_CONSTANT)
    axs[4].imshow(Iyy, cmap='gray')
    for i in range(0, len(cols)):
        axs[i].set(xlabel=cols[i])
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

