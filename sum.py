import argparse
import logging

from clstk.utils import colors

from clstk import linBilmes
from clstk import coRank
from clstk import simFusion

if __name__ == '__main__':
    common_parser = argparse.ArgumentParser(add_help=False)

    common_parser.add_argument('source_directory',
                               help='Directory containing a set of files to '
                               'be summarized.')
    common_parser.add_argument('-v', '--verbose', action='store_true',
                               help='Show verbose information messages')
    common_parser.add_argument('--no-colors', action='store_true',
                               help='Don\'t show colors in verbose log')

    parser = argparse.ArgumentParser(
            description='Automatically summarize a set of documents'
        )
    subparsers = parser.add_subparsers(title='methods',
                                       description='Summarization method')

    linBilmes.setupArgparse(subparsers.add_parser('linBilmes',
                                                  parents=[common_parser]))
    coRank.setupArgparse(subparsers.add_parser('coRank',
                                               parents=[common_parser]))
    simFusion.setupArgparse(subparsers.add_parser('simFusion',
                                                  parents=[common_parser]))

    args = parser.parse_args()

    if args.no_colors:
        colors.disable()

    logLevel = logging.NOTSET if args.verbose else logging.WARNING
    logging.basicConfig(
        level=logLevel,
        format=(colors.enclose('%(asctime)s', colors.CYAN) +
                colors.enclose('.%(msecs)03d ', colors.BLUE) +
                colors.enclose('%(name)s', colors.YELLOW) + ': ' +
                '%(message)s'),
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info("Initializing summarizer")

    args.func(args)
