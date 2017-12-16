from ..objectives import utils


def getParser():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        '-s', '--size', type=int, default=665, metavar="N",
        help='Maximum size of the summary')

    utils.addObjectiveParams(parser)

    return parser


def getParams(args=None):
    if not args:
        args = getParser().parse_args([])

    params = {
        'objectives': utils.getParams(args),
        'size': args.size
    }

    return params
