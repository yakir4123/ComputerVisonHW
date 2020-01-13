# import HW2.main as main
import HW3.main as main
# import HW4.main as main

import datetime


if __name__ == '__main__':
    """ start the imprted main """
    start = datetime.datetime.now()
    main.main()
    print("\033[94m\nTotal Running time {}".format(datetime.datetime.now() - start))
