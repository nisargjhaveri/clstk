import sys

from corpus import Corpus
from coverageObjective import CoverageObjective
from optimizer import Optimizer


def summarize(inDir):
    c = Corpus(inDir)
    c.load()

    optimizer = Optimizer()
    summary = optimizer.greedy(667, CoverageObjective(1), c)

    return summary


if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit("Usage: %s <input_directory>" % sys.argv[0])

    inDir = sys.argv[1]

    summary = summarize(inDir)
    print summary.getFormattedSummary()
