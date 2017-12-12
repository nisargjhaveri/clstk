from corpus import Corpus
from optimizer import Optimizer

from objectives import CoverageObjective
from objectives import DiversityRewardObjective
from objectives import AggregateObjective

import logging
logger = logging.getLogger("summarizer.py")


def summarize(inDir, params):
    logger.info("Loading documents from %s", inDir)
    c = Corpus(inDir).load()

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
    summary = optimizer.greedy(params["size"], objective, c)

    return summary
