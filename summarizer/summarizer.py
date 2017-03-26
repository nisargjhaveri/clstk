from corpus import Corpus
from optimizer import Optimizer
from coverageObjective import CoverageObjective
from dievrsityRewardObjective import DiversityRewardObjective
from aggregateObjective import AggregateObjective

import logging
logger = logging.getLogger("summarizer.py")


def summarize(inDir, params):
    logger.info("Loading documents from %s", inDir)
    c = Corpus(inDir)
    c.load()

    logger.info("Setting up summarizer")
    objective = AggregateObjective()

    if params["coverage"]["lambda"] > 0:
        objective.addObjective(
            params["coverage"]["lambda"],
            CoverageObjective(params["coverage"]["alphaN"])
        )

    if params["diversity"]["lambda"] > 0:
        objective.addObjective(
            params["diversity"]["lambda"],
            DiversityRewardObjective(params["diversity"]["kN"])
        )

    optimizer = Optimizer()
    summary = optimizer.greedy(665, objective, c)

    return summary
