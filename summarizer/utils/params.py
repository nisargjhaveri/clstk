from ..objectives import utils


def getParser():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        '-s', '--size', type=int, default=665, metavar="N",
        help='Maximum size of the summary')

    parser.add_argument(
        '-w', '--words', action="store_true",
        help='Caluated size as number of words instead of characters')

    parser.add_argument(
        '-l', '--target-lang', type=str, default=None, metavar="lang",
        help='Two-letter language code to generate cross-lingual summary. ' +
             'Source language is assumed to be English.')

    utils.addObjectiveParams(parser)

    return parser


def getParams(args=None):
    if not args:
        args = getParser().parse_args([])

    params = {
        'objectives': utils.getParams(args),
        'size': (args.size, args.words),
        'sourceLang': 'en',
        'targetLang': args.target_lang or 'en'
    }

    return params
