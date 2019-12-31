import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import math


def prob(site, image, temp):
    sum_eta = image[site[0], site[1] + 1] + image[site[0], site[1] - 1]\
              + image[site[0] - 1, site[1]] + image[site[0] + 1, site[1]]
    # math has better performance
    e = math.exp(2 * sum_eta / temp)  # with some calculation we can see its the same as gibbs for ising
    p_m1 = 1/(e + 1)
    return 1 if np.random.rand() > p_m1 else -1


def gibbs_sampling(temp, size, sweeps, image=None):
    bordered_image = np.zeros(np.array([2, 2]) + size)
    bordered_image[1:-1, 1:-1] = image if image is not None else np.random.randint(low=0, high=2, size=size)*2-1
    for _ in range(sweeps):
        for i in range(1, size[0] + 1):
            for j in range(1, size[1] + 1):
                bordered_image[i, j] = prob((i, j), bordered_image, temp)
    return bordered_image[1:-1, 1:-1]


def method1(times, size, sweeps, temp):
    sum_x12 = 0
    sum_x18 = 0
    for _ in range(times):
        sample = gibbs_sampling(temp, size, sweeps)
        sum_x12 += (sample[0][0] * sample[1][1])
        sum_x18 += (sample[0][0] * sample[-1][-1])
    print("Independent Samples:: For temp = {temp} => E(X11 x X22) = {res}".format(temp=temp, res=sum_x12/times))
    print("Independent Samples:: For temp = {temp} => E(X11 x X{n}{n}) = {res}".format(temp=temp, n=size[0], res=sum_x18/times))


def method2(sweeps, ignore_sweeps, size, temp):
    initial_image = gibbs_sampling(temp, size, ignore_sweeps)
    sweeps = sweeps - ignore_sweeps
    sample = gibbs_sampling(temp, size, sweeps, initial_image)
    Ex11x22 = sample[0][0] * sample[1][1]
    Ex11x88 = sample[0][0] * sample[-1][-1]
    print("Ergodicity:: For temp = {temp} => E(X11 x X22) = {res}".format(temp=temp, res=Ex11x22))
    print("Ergodicity:: For temp = {temp} => E(X11 x X{n}{n}) = {res}".format(temp=temp, n=size[0], res=Ex11x88))


def main():
    size = (8, 8)
    times = 10000
    temps = (1, 1.5, 2)

    for temp in temps:
        method1(times, size, 25, temp)
        method2(25000, 100, size, temp)
