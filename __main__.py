import sys

from corpus import Corpus
from optimizer import Optimizer
from coverageObjective import CoverageObjective
from aggregateObjective import AggregateObjective


def summarize(inDir):
    c = Corpus(inDir)
    c.load()

    objective = AggregateObjective()
    objective.addObjective(1, CoverageObjective(1))

    optimizer = Optimizer()
    summary = optimizer.greedy(667, objective, c)

    return summary


if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit("Usage: %s <input_directory>" % sys.argv[0])

    inDir = sys.argv[1]

    summary = summarize(inDir)
    print summary.getFormattedSummary()
