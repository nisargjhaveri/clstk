BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
END = '\033[0m'

disableColors = False


def enclose(str, color):
    return color + str + END if not disableColors else str


def isDisabled():
    return disableColors


def disable():
    global disableColors
    disableColors = True


def enable():
    global disableColors
    disableColors = False
