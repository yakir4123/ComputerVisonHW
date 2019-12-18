import multiprocessing

import numpy as np
import itertools
import matplotlib.pyplot as plt
import datetime

from cliqeFunctions import F, G, y2row


def T1(y2, temp, lattice_n):
    """
    Calculate T1(y2)
    :param y2: The argument that need to be calculated on
    :param temp: The temperature of the Ising model
    :param lattice_n: represent the nxn lattice: lattice_n x lattice_n
    :return: scalar of T1(y2)
    """
    y2_row = y2row(y2, lattice_n)
    sigma = lambda y1: G(y2row(y1, lattice_n), temp) * F(y2row(y1, lattice_n), y2_row, temp)
    return sum(map(sigma, range(0, 2 ** lattice_n)))


def Tn(Tn_minus1, temp, lattice_n):
    """
    Calculate Tn ( = Z_temp)
    :param Tn_minus1: List that already calculated T_{n-1} for every y_n
    :param temp: The temperature of the Ising model
    :param lattice_n: represent the nxn lattice: lattice_n x lattice_n
    :return: scalar of Tn
    """
    sigma = lambda yn: Tn_minus1[yn] * G(y2row(yn, lattice_n), temp)
    return sum(map(sigma, range(0, 2 ** lattice_n)))


def Tk(yk_plus1, Tk_minus1, temp, lattice_n):
    """
    :param yk_plus1:  The argument that need to be calculated
    :param Tk_minus1: List with all results that T_{k-1} can get
    :param temp: The temperature of the Ising model
    :param lattice_n: represent the nxn lattice: lattice_n x lattice_n
    :return: scalar of Tk(y_{k+1})
    """
    yk1_row = y2row(yk_plus1, lattice_n)
    sigma = lambda yk: Tk_minus1[yk] * \
                       G(y2row(yk, lattice_n), temp) * \
                       F(y2row(yk, lattice_n), yk1_row, temp)
    return sum(map(sigma, range(0, 2 ** lattice_n)))


def calculateT(temp, lattice_n):
    """
    Calculate using iteration T to calculate the distribution later.
    :param temp: The temperature of the Ising model
    :param lattice_n: represent the nxn lattice: lattice_n x lattice_n
    :return: List of [Tn .. T1], every Tk is a list with all values that it can get except of Tn that its scalar
    """
    T_1 = [T1(y2, temp, lattice_n) for y2 in range(0, 2 ** lattice_n)]

    # list with T1..T_{n-1}
    T_ks = list(itertools.accumulate([T_1] + list(range(1, lattice_n - 1)),
                                     lambda Tk_minus1, _: [Tk(yk_plus1, Tk_minus1, temp, lattice_n)
                                                           for yk_plus1 in range(0, 2 ** lattice_n)]))
    T_n = Tn(T_ks[-1], temp, lattice_n)
    # Concat the reuslts plus our only need of T is for P and to calculate it we do it backwards
    return (T_ks + [T_n])


def calculate_distribution(T, temp):
    """
    calculate the distribution table
    :param T: The algorithm's T to calculate P_{i|i-1)
    :param Y: List Y = [y1 .. yn]
    :return:
    """
    n = len(T)
    # Pn = T_{n-1}(yn) * G(yn) / Tn
    Pn = [T[n-2][Yn] * G(y2row(Yn, n), temp) / T[n-1]
          for Yn in range(0, 2**n)]
    # list of all Pk where k in [1.. n-2]  ( or [2 .. n-1] if you look on the algorithm )
    Pk = [[[T[k-1][Yk] * G(y2row(Yk, n), temp) * F(y2row(Yk, n), y2row(Yk_plus1, n), temp) / T[k][Yk_plus1]
          for Yk in range(0, 2**n)]
          for Yk_plus1 in range(0, 2**n)]
          for k in range(n - 2, 0, -1)]
    P1 = [[G(y2row(Y0, n), temp) * F(y2row(Y0, n), y2row(Y1, n), temp) / T[0][Y1]
          for Y0 in range(0, 2**n)]
          for Y1 in range(0, 2**n)]
    return [Pn] + Pk + [P1]


def sampleIsing(P):
    """
    Sample Ising image from P
    :param P: pmf and condition pmf for each Y sorted from n to 1
    :return: Ising model represented Y
    """
    n = len(P)
    yn = np.random.choice(np.arange(0, 2**n), p=P[0])
    yk = itertools.accumulate([yn] + list(range(1, n)),
                              lambda k_plus1, k: np.random.choice(np.arange(0, 2**n), p=P[k][k_plus1]))
    return yk


def generateImages(temp, samples, lattice_n):
    T = calculateT(temp, lattice_n)
    P = calculate_distribution(T, temp)
    for _ in range(0,samples):
        yield sampleIsing(P)


def worker(temp, samples, lattice_n):
    """
    worker for pool create list of 'samples' images lattice_n x lattice_n
    """
    return [[y2row(yi, lattice_n) for yi in Y] for Y in generateImages(temp, samples, lattice_n)]


def computer_exercise7(lattice_n=8):
    temps = (1, 1.5, 2)
    samples = 10
    fig, axs = plt.subplots(len(temps), samples)

    cols = ['{}'.format(col) for col in range(1, 11)]
    rows = ['Temp {}'.format(temp) for temp in temps]

    for i in range(0, samples):
        axs[len(temps) - 1, i].set(xlabel=cols[i])
    for i in range(0, len(temps)):
        axs[i, 0].set(ylabel=rows[i])
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    pool = multiprocessing.Pool(processes=len(temps))
    # call with different process to worker(temp,samples,lattice_n)
    images = pool.starmap(worker, map(lambda temp: (temp, samples, lattice_n) , temps))
    [axs[i, j].imshow(images[i][j], cmap='gray', vmin=-1, vmax=1) for j in range(0, samples) for i in range(0, len(images))]
    plt.show()


def main():
    start = datetime.datetime.now()
    computer_exercise7(8)
    print(datetime.datetime.now() - start)
