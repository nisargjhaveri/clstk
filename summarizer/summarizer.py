from corpus import Corpus
from optimizer import Optimizer
from coverageObjective import CoverageObjective
from dievrsityRewardObjective import DiversityRewardObjective
from aggregateObjective import AggregateObjective

import logging
logger = logging.getLogger("summarizer.py")


def summarize(inDir):
    logger.info("Loading documents from %s", inDir)
    c = Corpus(inDir)
    c.load()

    logger.info("Setting up summarizer")
    objective = AggregateObjective()
    objective.addObjective(1, CoverageObjective(5))
    objective.addObjective(6, DiversityRewardObjective(0.1))

    optimizer = Optimizer()
    summary = optimizer.greedy(665, objective, c)

    return summary
