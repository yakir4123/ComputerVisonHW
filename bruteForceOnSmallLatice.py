import numpy as np
from cliqeFunctions import F, G


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


def wrapper_print(exercise_number, temps, exercise, lattice_width):
    print("\n\t--- Computer Exercise {} ---".format(exercise_number))
    for temp_ in temps:
        print("Z_{temp} of {lat}x{lat} lattice = {ex}".format(temp=temp_, lat=lattice_width, ex=exercise(temp_)))
    print("\t--- Computer Exercise {} ---".format(exercise_number))


def main():
    print("Brute Force On Small Lattice")
    temps = (1, 1.5, 2)
    wrapper_print(3, temps, computer_exercise3, 2)
    wrapper_print(4, temps, computer_exercise4, 3)
    wrapper_print(5, temps, computer_exercise5, 2)
    wrapper_print(6, temps, computer_exercise6, 3)
