import sys
import os

import utils
from corpus import Corpus
from coverageObjective import CoverageObjective
from optimizer import Optimizer

if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit("Usage: %s <input_directory> <output_directory>" % sys.argv[0])

    inDir = sys.argv[1]
    outDir = sys.argv[2]

    tmpDir = os.path.join(outDir, "tmp")

    utils.ensureDir(outDir)
    utils.ensureDir(tmpDir)

    c = Corpus(inDir)
    c.load()

    optimizer = Optimizer()
    summary = optimizer.greedy(667, CoverageObjective(1), c)

    print summary.getFormattedSummary()
