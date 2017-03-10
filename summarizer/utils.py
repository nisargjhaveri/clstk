import os


def ensureDir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class colors:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    END = '\033[0m'

    disableColors = False

    @staticmethod
    def enclose(str, color):
        return color + str + colors.END if not colors.disableColors else str

    @staticmethod
    def isDisabled():
        return colors.disableColors

    @staticmethod
    def disable():
        colors.disableColors = True

    @staticmethod
    def enable():
        colors.disableColors = False
