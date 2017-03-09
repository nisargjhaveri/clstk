from corpus import Corpus
from optimizer import Optimizer
from coverageObjective import CoverageObjective
from dievrsityRewardObjective import DiversityRewardObjective
from aggregateObjective import AggregateObjective


def summarize(inDir):
    c = Corpus(inDir)
    c.load()

    objective = AggregateObjective()
    # objective.addObjective(1, CoverageObjective(5))
    objective.addObjective(1, DiversityRewardObjective(0.2))

    optimizer = Optimizer()
    summary = optimizer.greedy(665, objective, c)

    return summary
