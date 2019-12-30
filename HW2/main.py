import bruteForceOnSmallLatice as bruteForceOnSmallLatice
import dynamicProgramming as dynamicProgramming

import datetime


def main():
    bruteForceOnSmallLatice.main()
    dynamicProgramming.main()


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print("\033[94m\nTotal Running time {}".format(datetime.datetime.now() - start))
