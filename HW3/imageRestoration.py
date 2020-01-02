from HW3.estimationNearlyExactValue import gibbs_sampling, ising_prior_pmf

import numpy as np
import math
import matplotlib.pyplot as plt


def ising_posterior_pmf(site, temp, x, y, sq_sigma):
    sum_eta = x[site[0], site[1] + 1] + x[site[0], site[1] - 1] \
              + x[site[0] - 1, site[1]] + x[site[0] + 1, site[1]]
    # with calculation we can get a less computional pmf
    # y is 2 degrees less than x because it wont have borders
    e = math.exp(2 * (sum_eta / temp + (y[site[0] - 1, site[1] - 1] / sq_sigma)))
    p_m1 = 1 / (e + 1)
    return 1 if np.random.rand() > p_m1 else -1


def argmax_ising_posterior(site, temp, x, y, sq_sigma):
    sum_eta = x[site[0], site[1] + 1] + x[site[0], site[1] - 1] \
              + x[site[0] - 1, site[1]] + x[site[0] + 1, site[1]]
    # with calculation we can get a less computional pmf
    # y is 2 degrees less than x because it wont have borders
    e = math.exp(2 * (sum_eta / temp + (y[site[0] - 1, site[1] - 1] / sq_sigma)))
    p_m1 = 1 / (e + 1)
    return 1 if p_m1 < 1 - p_m1 else -1


def main():
    size = (100, 100)
    temps = (1, 1.5, 2)
    steps = 5
    fig, axs = plt.subplots(len(temps), steps)
    cols = ['prior sample', 'noised sample', 'posterior sample', 'ICM', 'maximum estimator']
    rows = ['Temp {}'.format(temp) for temp in temps]

    for i in range(0, steps):
        axs[len(temps) - 1, i].set(xlabel=cols[i])
    for i in range(0, len(temps)):
        axs[i, 0].set(ylabel=rows[i])
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    for temp in temps:
        # step 1, sample 8x8 image using gibbs sampler
        prior_sample = gibbs_sampling(temp, size, 50, ising_prior_pmf)
        axs[temps.index(temp), 0].imshow(prior_sample, cmap='gray')

        # step 2, add noise
        sigma = 2
        eta = np.random.normal(0, sigma, size)
        noised_sample = prior_sample + eta
        axs[temps.index(temp), 1].imshow(noised_sample, cmap='gray')

        # step 3, posterior sample
        # add zero borders to handle neighbors of the borders
        border_noised_sample = np.zeros(np.array([2, 2]) + size)
        border_noised_sample[1:-1, 1:-1] = noised_sample
        posterior_sample = gibbs_sampling(temp, size, 50, ising_posterior_pmf, None, noised_sample, sigma * sigma)
        axs[temps.index(temp), 2].imshow(posterior_sample, cmap='gray')

        # step 4
        argmax_posterior_sample = gibbs_sampling(temp, size, 50, argmax_ising_posterior, None, noised_sample, sigma * sigma)
        axs[temps.index(temp), 3].imshow(argmax_posterior_sample, cmap='gray')

        # step 5
        argmax_likelihood = np.sign(noised_sample)
        axs[temps.index(temp), 4].imshow(argmax_likelihood, cmap='gray')

    plt.show()
