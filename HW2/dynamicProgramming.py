import multiprocessing

import numpy as np
import itertools
import matplotlib.pyplot as plt

from HW2.cliqeFunctions import F, G, y2row


class Experiment:

    def __init__(self, temp, lattice_n):
        self.temp = temp
        self.lattice_n = lattice_n
        self.y2rows = dict()
        self.P = []
        self.calculate_distribution()

    def y2row(self, y):
        if y not in self.y2rows:
            self.y2rows[y] = y2row(y, self.lattice_n)
        return self.y2rows[y]

    def T1(self, y2):
        """
        Calculate T1(y2)
        :param y2: The argument that need to be calculated on
        :param temp: The temperature of the Ising model
        :param lattice_n: represent the nxn lattice: lattice_n x lattice_n
        :return: scalar of T1(y2)
        """
        temp = self.temp
        y2_row = self.y2row(y2)
        sigma = lambda y1: G(self.y2row(y1), temp) * F(self.y2row(y1), y2_row, temp)
        return sum(map(sigma, range(0, 2 ** self.lattice_n)))

    def Tn(self, Tn_minus1):
        """
        Calculate Tn ( = Z_temp)
        :param Tn_minus1: List that already calculated T_{n-1} for every y_n
        :param temp: The temperature of the Ising model
        :param lattice_n: represent the nxn lattice: lattice_n x lattice_n
        :return: scalar of Tn
        """
        sigma = lambda yn: Tn_minus1[yn] * G(self.y2row(yn), self.temp)
        return sum(map(sigma, range(0, 2 ** self.lattice_n)))

    def Tk(self, yk_plus1, Tk_minus1):
        """
        :param yk_plus1:  The argument that need to be calculated
        :param Tk_minus1: List with all results that T_{k-1} can get
        :param temp: The temperature of the Ising model
        :param lattice_n: represent the nxn lattice: lattice_n x lattice_n
        :return: scalar of Tk(y_{k+1})
        """
        temp = self.temp
        yk1_row = self.y2row(yk_plus1)
        sigma = lambda yk: Tk_minus1[yk] * \
                           G(self.y2row(yk), temp) * \
                           F(self.y2row(yk), yk1_row, temp)
        return sum(map(sigma, range(0, 2 ** self.lattice_n)))

    def calculateT(self):
        """
        Calculate using iteration T to calculate the distribution later.
        :param temp: The temperature of the Ising model
        :param lattice_n: represent the nxn lattice: lattice_n x lattice_n
        :return: List of [Tn .. T1], every Tk is a list with all values that it can get except of Tn that its scalar
        """
        lattice_n = self.lattice_n
        T_1 = [self.T1(y2) for y2 in range(0, 2 ** lattice_n)]

        # list with T1..T_{n-1}
        T_ks = list(itertools.accumulate([T_1] + list(range(1, lattice_n - 1)),
                                         lambda Tk_minus1, _: [self.Tk(yk_plus1, Tk_minus1)
                                                               for yk_plus1 in range(0, 2 ** lattice_n)]))
        T_n = self.Tn(T_ks[-1])
        # Concat the reuslts plus our only need of T is for P and to calculate it we do it backwards
        return T_ks + [T_n]

    def calculate_distribution(self):
        """
        calculate the distribution table
        """
        lattice_n = self.lattice_n
        temp = self.temp
        T = self.calculateT()
        n = lattice_n
        # Pn = T_{n-1}(yn) * G(yn) / Tn
        Pn = np.fromiter((T[n - 2][Yn] * G(self.y2row(Yn), temp) / T[n - 1]
                         for Yn in range(0, 2 ** n)), dtype='float64')
        # list of all Pk where k in [1.. n-2]  ( or [2 .. n-1] if you look on the algorithm )
        Pk = [np.array([
            np.fromiter((T[k - 1][Yk] * G(self.y2row(Yk), temp) * F(self.y2row(Yk), self.y2row(Yk_plus1), temp) / T[k][Yk_plus1]
                        for Yk in range(0, 2 ** n)), dtype='float64')
                            for Yk_plus1 in range(0, 2 ** n)])
                                for k in range(n - 2, 0, -1)]
        P1 = np.array([np.fromiter((G(self.y2row(Y0), temp) * F(self.y2row(Y0), self.y2row(Y1), temp) / T[0][Y1]
                        for Y0 in range(0, 2 ** n)), dtype='float64')
                            for Y1 in range(0, 2 ** n)])
        # self.P = [Pn] + Pk + [P1]
        Pk.reverse()
        self.P = [Pn] + Pk + [P1]
        return self.P


    def sampleIsing(self):
        """
        Sample Ising image from P
        :param P: pmf and condition pmf for each Y sorted from n to 1
        :return: Ising model represented Y
        """
        n = len(self.P)
        yn = np.random.choice(np.arange(0, 2 ** n), p=self.P[0])
        yk = itertools.accumulate([yn] + list(range(1, n)),
                                  lambda k_plus1, k: np.random.choice(np.arange(0, 2 ** n), p=self.P[k][k_plus1]))
        return yk

    def generateImages(self, samples):
        for _ in range(0, samples):
            yield self.sampleIsing()


def worker(experiment, samples):
    """
    worker for pool create list of 'samples' images lattice_n x lattice_n
    """
    return [[y2row(yi, experiment.lattice_n) for yi in Y] for Y in experiment.generateImages(samples)]


def computer_exercise7(experiments):
    samples = 10
    fig, axs = plt.subplots(len(experiments), samples)

    cols = ['{}'.format(col) for col in range(1, 11)]
    rows = ['Temp {}'.format(exp.temp) for exp in experiments]

    for i in range(0, samples):
        axs[len(experiments) - 1, i].set(xlabel=cols[i])
    for i in range(0, len(experiments)):
        axs[i, 0].set(ylabel=rows[i])
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    pool = multiprocessing.Pool(processes=len(experiments))
    # call with different process to worker(temp,samples,lattice_n)
    images = pool.starmap(worker, map(lambda exp: (exp, samples), experiments))
    [axs[i, j].imshow(images[i][j], cmap='gray', vmin=-1, vmax=1, interpolation="None")
        for j in range(0, samples) for i in range(0, len(images))]
    pool.close()
    plt.show()


def computer_exercise8(experiments):
    samples = 10000
    slices = 1
    pool = multiprocessing.Pool(processes=len(experiments)*slices)
    work_on = list(itertools.chain(*[[(exp, samples//slices) for _ in range(0, slices) for exp in experiments]]))
    work_on.sort(key=lambda tup: tup[0].temp)
    workers_images = pool.starmap(worker, work_on)
    pool.close()
    # workers_images = [worker(experiment, samples) for experiment in experiments]
    for i in range(0, len(experiments)):
        res = sum(image[0][0] * np.array((image[1][1], image[-1][-1]))
                  for slice in workers_images[i:i+slices] for image in slice)/samples
        print("For temp = {temp} => E(X11 x X22) = {res}".format(temp=experiments[i].temp, res=res[0]))
        print("For temp = {temp} => E(X11 x X{n}{n}) = {res}".format(temp=experiments[i].temp,
                                                                     n=experiments[i].lattice_n, res=res[1]))


def create_experiment(temp, lattice_n):
    return Experiment(temp, lattice_n)


def main():
    temps = (1, 1.5, 2)
    lattice_n = 8

    pool = multiprocessing.Pool(processes=len(temps))
    experiments = pool.starmap(create_experiment, map(lambda temp: (temp, lattice_n), temps))
    # computer_exercise7(experiments)
    print("\n\t--- Computer Exercise 8 ---")
    computer_exercise8(experiments)
    print("\t--- Computer Exercise 8 ---")
    pool.close()
