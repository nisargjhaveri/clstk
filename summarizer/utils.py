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


class Params(object):
    @staticmethod
    def getParser():
        import argparse
        parser = argparse.ArgumentParser(add_help=False)

        parser.add_argument(
            '-s', '--size', type=int, default=665, metavar="N",
            help='Maximum size of the summary')

        parser.add_argument(
            '--coverage-lambda', type=float, default=1.0, metavar="lambda",
            help='Coefficient for coverage objective')
        parser.add_argument(
            '--coverage-alpha', type=float, default=6.0, metavar="alphaN",
            help='Threshold co-efficient to be used in coverage objective.'
            + ' The co-efficient the  will be calucated as alphaN / N')

        parser.add_argument(
            '--diversity-lambda', type=float, default=6.0, metavar="lambda",
            help='Coefficient for diversity objective')
        parser.add_argument(
            '--diversity-k', type=float, default=0.1, metavar="kN",
            help='Number of clusters for diversity objective.'
            + ' Number of clustres will be calucated as kN * N')

        return parser

    @staticmethod
    def getParams(args=None):
        if not args:
            args = Params.getParser().parse_args([])

        params = {
            'coverage': {
                'lambda': args.coverage_lambda,
                'alphaN': args.coverage_alpha,
            },
            'diversity': {
                'lambda': args.diversity_lambda,
                'kN': args.diversity_k,
            },
            'size': args.size
        }

        return params
