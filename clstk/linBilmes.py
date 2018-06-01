from corpus import Corpus
from summary import Summary

import objectives
from objectives import AggregateObjective

import logging
logger = logging.getLogger("linBilmes.py")


def optimizeGreedy(sizeBudget, objective, corpus):
    summary = Summary()
    sentencesLeft = corpus.getSentences()

    objective.setCorpus(corpus)

    sizeBudget, countTokens = sizeBudget
    sizeName = "tokens" if countTokens else "chars"

    def sentenceSize(sent):
        return sent.tokenCount() if countTokens else sent.charCount()

    def summarySize(summary):
        return summary.tokenCount() if countTokens else summary.charCount()

    logger.info("Greedily optimizing the objective")
    logger.info("Summary budget: %d %s", sizeBudget, sizeName)
    while summarySize(summary) < sizeBudget and len(sentencesLeft) > 0:
        objectiveValues = map(objective.getObjective(summary),
                              sentencesLeft)
        maxObjectiveValue = max(objectiveValues)

        candidates = [sentencesLeft[i]
                      for i, v in enumerate(objectiveValues)
                      if v == maxObjectiveValue]

        candidateSizes = map(sentenceSize, candidates)
        minSize = min(candidateSizes)

        selectedCandidate = candidates[candidateSizes.index(minSize)]
        sentencesLeft.remove(selectedCandidate)

        if summarySize(summary) + minSize <= sizeBudget:
            logger.info("Sentence added with objective value: %f, " +
                        "size: %d", maxObjectiveValue, minSize)
            summary.addSentence(selectedCandidate)

        budgetLeft = sizeBudget - summarySize(summary)
        sentencesLeft = filter(lambda s: sentenceSize(s) < budgetLeft,
                               sentencesLeft)

    logger.info("Optimization done, summary size: %d chars, %d tokens",
                summary.charCount(), summary.tokenCount())

    return summary


def summarize(inDir, params):
    logger.info("Loading documents from %s", inDir)
    c = Corpus(inDir).load(
            params,
            translate=True,
            replaceWithTranslation=params['earlyTranslate'],
            simplify=(params['simplify'] is not None),
            replaceWithSimplified=(params['simplify'] == 'early'),
        )

    logger.info("Setting up summarizer")
    objective = AggregateObjective(params['objectives'])

    summary = optimizeGreedy(params["size"], objective, c)

    return summary


def setupArgparse(parser):
    def run(args, silent=False):
        params = {
            'objectives': objectives.utils.getParams(args),
            'size': (args.size, args.words),
            'sourceLang': args.source_lang,
            'targetLang': args.target_lang or args.source_lang,
            'earlyTranslate': args.early_translate,
            'simplify': (args.simplify
                         if args.simplify in ['early', 'late'] else None)
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
    parser.add_argument('--source-lang', type=str, default='en',
                        metavar="lang", help='Two-letter language code of '
                        'the source documents language. Defaults to `en`')
    parser.add_argument('-l', '--target-lang', type=str, default=None,
                        metavar="lang", help='Two-letter language code to '
                        'generate cross-lingual summary. '
                        'Defaults to source language.')
    parser.add_argument('--early-translate', action="store_true",
                        help='First translate and then summarize.')
    parser.add_argument('--simplify', type=str, default='never',
                        choices=['early', 'never'],
                        help='When to simplify sentences and then summarize.')

    objectives.utils.addObjectiveParams(parser)

    parser.set_defaults(func=run)
