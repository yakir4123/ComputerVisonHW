import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import math


def ising_prior_pmf(site, temp, x):
    sum_eta = x[site[0], site[1] + 1] + x[site[0], site[1] - 1] \
              + x[site[0] - 1, site[1]] + x[site[0] + 1, site[1]]

    # math has better performance than numpy
    e = math.exp(2 * sum_eta / temp)
    # with some calculation we can see its the same as gibbs for prior ising
    # has better performance
    p_m1 = 1/(e + 1)
    return 1 if np.random.rand() > p_m1 else -1


def gibbs_sampling(temp, size, sweeps, pmf, x=None, *known_rv):
    bordered_x = np.zeros(np.array([2, 2]) + size)
    bordered_x[1:-1, 1:-1] = x if x is not None else np.random.randint(low=0, high=2, size=size) * 2 - 1
    for _ in range(sweeps):
        for i in range(1, size[0] + 1):
            for j in range(1, size[1] + 1):
                bordered_x[i, j] = pmf((i, j), temp, bordered_x, *known_rv)
    return bordered_x[1:-1, 1:-1]


def method1(times, size, sweeps, temp):
    sum_x12 = 0
    sum_x18 = 0
    for _ in range(times):
        sample = gibbs_sampling(temp, size, sweeps, ising_prior_pmf)
        sum_x12 += (sample[0][0] * sample[1][1])
        sum_x18 += (sample[0][0] * sample[-1][-1])
    print("Independent Samples:: For temp = {temp} => E(X11 x X22) = {res}".format(temp=temp, res=sum_x12/times))
    print("Independent Samples:: For temp = {temp} => E(X11 x X{n}{n}) = {res}".format(temp=temp, n=size[0], res=sum_x18/times))


def method2(sweeps, ignore_sweeps, size, temp):
    initial_image = gibbs_sampling(temp, size, ignore_sweeps, ising_prior_pmf)
    sweeps = sweeps - ignore_sweeps
    sum_x12 = 0
    sum_x18 = 0
    sample = initial_image
    for _ in range(0, sweeps - ignore_sweeps):
        sample = gibbs_sampling(temp, size, 1, ising_prior_pmf, sample)
        sum_x12 += sample[0][0] * sample[1][1]
        sum_x18 += sample[0][0] * sample[-1][-1]
    print("Ergodicity:: For temp = {temp} => E(X11 x X22) = {res}"
          .format(temp=temp, res=sum_x12 / (sweeps - ignore_sweeps)))
    print("Ergodicity:: For temp = {temp} => E(X11 x X{n}{n}) = {res}"
          .format(temp=temp, n=size[0], res=sum_x18 / (sweeps - ignore_sweeps)))


def worker(times, size, sweeps, temp, sweeps_ignore):
    method1(times, size, sweeps, temp)
    method2(times * sweeps, sweeps_ignore, size, temp)


def main():
    size = (8, 8)
    times = 10000
    sweeps = 25
    temps = (1, 1.5, 2)
    sweeps_ignore = 100

    worker_args = [(times, size, sweeps, temp, sweeps_ignore) for temp in temps]
    with multiprocessing.Pool(len(temps)) as p:
        p.starmap(worker, worker_args)
