from corpus import Corpus
from optimizer import Optimizer

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
        summary.translateCollection(params['sourceLang'], params['targetLang'])

    return summary
