from corpus import Corpus
from optimizer import Optimizer

import objectives
from objectives import AggregateObjective

import logging
logger = logging.getLogger("summarizer.py")


def summarize(inDir, params):
    logger.info("Loading documents from %s", inDir)
    c = Corpus(inDir).load(params)

    logger.info("Setting up summarizer")
    objective = AggregateObjective(params['objectives'])

    optimizer = Optimizer()
    summary = optimizer.greedy(params["size"], objective, c)

    if params['sourceLang'] != params['targetLang']:
        logger.info("Translating summary")
        summary.translate(params['sourceLang'], params['targetLang'])

    return summary


def setupArgparse(parser):
    def run(args, silent=False):
        params = {
            'objectives': objectives.utils.getParams(args),
            'size': (args.size, args.words),
            'sourceLang': 'en',
            'targetLang': args.target_lang or 'en'
        }

        summary = summarize(args.source_directory, params)

        if not silent:
            logging.info("Printing source summary\n" +
                         summary.getSummary().encode('utf-8'))

            logging.info("Printing target summary")
            print summary.getTargetSummary().encode('utf-8')

        return summary

    parser.add_argument('-s', '--size', type=int, default=665, metavar="N",
                        help='Maximum size of the summary')
    parser.add_argument('-w', '--words', action="store_true",
                        help='Caluated size as number of words instead of '
                        'characters')
    parser.add_argument('-l', '--target-lang', type=str, default=None,
                        metavar="lang", help='Two-letter language code to '
                        'generate cross-lingual summary. '
                        'Source language is assumed to be English.')

    objectives.utils.addObjectiveParams(parser)

    parser.set_defaults(func=run)
