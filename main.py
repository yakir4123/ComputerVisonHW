# import HW2.main
# import HW3.main
import HW4.main


import datetime


if __name__ == '__main__':
    start = datetime.datetime.now()
    HW4.main.main()
    print("\033[94m\nTotal Running time {}".format(datetime.datetime.now() - start))
