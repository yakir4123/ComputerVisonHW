import HW2.bruteForceOnSmallLatice as bruteForceOnSmallLatice
import HW2.dynamicProgramming as dynamicProgramming

import datetime


def main():
    bruteForceOnSmallLatice.main()
    dynamicProgramming.main()


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print("\033[94m\nTotal Running time {}".format(datetime.datetime.now() - start))
