import numpy as np
import itertools
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


def calculateDisterbution(T, temp):
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
    return ([Pn] + Pk + [P1])[::-1]


def computer_exercise7(temp, lattice_n=8):
    T = calculateT(temp, lattice_n)
    P = calculateDisterbution(T, 1)
    print(10)


def main():
    computer_exercise7(1, 3)
