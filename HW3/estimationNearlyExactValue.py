import numpy as np


def prob(site, image, temp):
    sum_eta = image[site[0]][site[1] + 1] + image[site[0]][site[1] - 1] + image[site[0] - 1][site[1]] + image[site[0] + 1][site[1]]
    e = np.exp(2 * sum_eta / temp)  # with some calculation we can see its the same as gibbs for ising
    p_m1 = 1/(e + 1)
    return 1 if np.random.rand() > p_m1 else -1


def method1(temp, size, sweeps):
    bordered_image = np.zeros(np.array([2, 2]) + size)
    bordered_image[1:-1, 1:-1] = np.random.randint(low=0, high=2, size=size)*2-1
    for _ in range(sweeps):
        for i in range(1, size[0] + 1):
            for j in range(1, size[1] + 1):
                bordered_image[i, j] = prob((i, j), bordered_image, temp)
    return bordered_image[1:-1, 1:-1]


def main():
    size = (8, 8)
    times = 100
    sweeps = 25

    for temp in (1, 1.5, 2):
        sum_x12 = 0
        sum_x18 = 0
        for _ in range(times):
            sample = method1(temp, size, sweeps)
            sum_x12 += (sample[0][0] * sample[1][1])
            sum_x18 += (sample[0][0] * sample[7][7])
        print("For temp = {temp} => E(X11 x X22) = {res}".format(temp=temp, res=sum_x12/times))
        print("For temp = {temp} => E(X11 x X{n}{n}) = {res}".format(temp=temp, n=size[0], res=sum_x18/times))
