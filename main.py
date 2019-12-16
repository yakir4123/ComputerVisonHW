import numpy as np


def G(row_s, temp):
    return np.exp(np.sum(row_s[1:] * row_s[:-1]) / temp)


def F(row_s, row_t, temp):
    return np.exp(np.sum(row_s * row_t) / temp)


def computer_exercise3(temp):
    S = (-1, 1)
    sum_of_options = 0
    for x11 in S:
        for x12 in S:
            for x21 in S:
                for x22 in S:
                    option = np.exp((x11 * x12 +
                                 x12 * x22 +
                                 x22 * x21 +
                                 x21 * x11)/temp)
                    sum_of_options = sum_of_options + option
    return sum_of_options


def main():
    temp_exp = (1, 1.5, 2)
    for temp in temp_exp:
        print("Z_{0} of 2x2 lattice = {1}".format(temp, computer_exercise3(temp)))


if __name__ == '__main__':
    main()
