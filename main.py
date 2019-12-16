import numpy as np


def G(row_s, temp):
    return np.exp(np.sum(row_s[1:] * row_s[:-1]) / temp)


def main():
    pass


if __name__ == '__main__':
    main()
