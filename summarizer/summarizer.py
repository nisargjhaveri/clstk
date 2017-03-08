from corpus import Corpus
from optimizer import Optimizer
from coverageObjective import CoverageObjective
from aggregateObjective import AggregateObjective


def summarize(inDir):
    c = Corpus(inDir)
    c.load()

    objective = AggregateObjective()
    objective.addObjective(1, CoverageObjective(5))

    optimizer = Optimizer()
    summary = optimizer.greedy(665, objective, c)

    return summary
