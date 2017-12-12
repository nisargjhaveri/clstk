from corpus import Corpus
from optimizer import Optimizer

from objectives import AggregateObjective

import logging
logger = logging.getLogger("summarizer.py")


def summarize(inDir, params):
    logger.info("Loading documents from %s", inDir)
    c = Corpus(inDir).load()

    logger.info("Setting up summarizer")
    objective = AggregateObjective(params['objectives'])

    optimizer = Optimizer()
    summary = optimizer.greedy(params["size"], objective, c)

    return summary
