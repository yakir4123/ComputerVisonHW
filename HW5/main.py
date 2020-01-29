from HW5.imageWrapping import computer_exercise2, computer_exercise3
from HW5.opticalFlow import computer_exercise4, computer_exercise5


def main():
    computer_exercise2()
    computer_exercise3()
    computer_exercise4()
    for l in [0,0.1, 0.5, 1, 1.5]:
        computer_exercise5(lambda_=l)

