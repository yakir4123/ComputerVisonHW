import numpy as np


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


def computer_exercise4(temp):
    S = (-1, 1)
    sum_of_options = 0
    for x11 in S:
        for x12 in S:
            for x13 in S:
                for x21 in S:
                    for x22 in S:
                        for x23 in S:
                            for x31 in S:
                                for x32 in S:
                                    for x33 in S:
                                        option = np.exp((
                                            # rows
                                            x11 * x12 +
                                            x12 * x13 +
                                            x21 * x22 +
                                            x22 * x23 +
                                            x31 * x32 +
                                            x32 * x33 +
                                            # col
                                            x11 * x21 +
                                            x21 * x31 +
                                            x12 * x22 +
                                            x22 * x32 +
                                            x13 * x23 +
                                            x23 * x33)/temp)
                                        sum_of_options = sum_of_options + option
    return sum_of_options


def computer_exercise5(temp):
    S = range(0, 4)
    sum_of_options = 0
    for y1 in S:
        row_y1 = y2row(y1, 2)
        for y2 in S:
            row_y2 = y2row(y2, 2)
            sum_of_options = sum_of_options + (G(row_y1, temp) * G(row_y2, temp) * F(row_y1, row_y2, temp))
    return sum_of_options



def computer_exercise6(temp):
    S = range(0, 8)
    sum_of_options = 0
    for y1 in S:
        row_y1 = y2row(y1, 3)
        for y2 in S:
            row_y2 = y2row(y2, 3)
            for y3 in S:
                row_y3 = y2row(y3, 3)
                sum_of_options = sum_of_options + (G(row_y1, temp) * G(row_y2, temp) * G(row_y3, temp)
                                                   * F(row_y1, row_y2, temp) * F(row_y2, row_y3, temp))
    return sum_of_options


def main():
    temp_exp = (1, 1.5, 2)
    print("\n\t--- Computer Exercise 3 ---")
    for temp in temp_exp:
        print("Z_{0} of 2x2 lattice = {1}".format(temp, computer_exercise3(temp)))
    print("\n\t--- Computer Exercise 4 ---")
    for temp in temp_exp:
        print("Z_{0} of 3x3 lattice = {1}".format(temp, computer_exercise4(temp)))
    print("\n\t--- Computer Exercise 5 ---")
    for temp in temp_exp:
        print("Z_{0} of 2x2 lattice = {1}".format(temp, computer_exercise5(temp)))
    print("\n\t--- Computer Exercise 6 ---")
    for temp in temp_exp:
        print("Z_{0} of 3x3 lattice = {1}".format(temp, computer_exercise6(temp)))


if __name__ == '__main__':
    main()
