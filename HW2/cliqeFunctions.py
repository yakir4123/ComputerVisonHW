import numpy as np


def G(row_s, temp):
    return np.exp(np.sum(row_s[1:] * row_s[:-1]) / temp)


def F(row_s, row_t, temp):
    return np.exp(np.sum(row_s * row_t) / temp)


def y2row(y, width=8):
    """
    y: an integer in (0,...,(2**width)-1)
    """
    if not 0 <= y <= (2**width)-1:
        raise ValueError(y)
    my_str = np.binary_repr(y, width)
    my_list = list(map(int, my_str))
    my_array = np.asarray(my_list)
    my_array[my_array == 0] = -1
    row = my_array
    return row
