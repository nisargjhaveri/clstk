import argparse
import logging

from utils import Params
from utils import colors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            parents=[Params.getParser()],
            description='Automatically summarize a set of documents'
        )
    parser.add_argument('source_directory',
                        help='Directory containing a set of files to be ' +
                        'summarized.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show verbose information messages')
    parser.add_argument('--no-colors', action='store_true',
                        help='Don\'t show colors in verbose log')

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

    from summarizer import summarize

    summary = summarize(args.source_directory, Params.getParams(args))
    print summary.getFormattedSummary()
