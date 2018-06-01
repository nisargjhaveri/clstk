import os
import subprocess
import tempfile
import math
import shutil

import sklearn.metrics.pairwise
import numpy as np

from ..utils.param import Param
from ._objective import Objective

import logging
logger = logging.getLogger("diversityRewardObjective.py")


class DiversityRewardObjective(Objective):
    def __init__(self, params):
        self.kN = params['k']

    @staticmethod
    def getParams():
        return [
            Param(
                'k', type=float, default=0.1, metavar="kN",
                help='Number of clusters for diversity objective.'
                + ' Number of clustres will be calucated as kN * N'
            )
        ]

    def _executeCLUTO(self, matrixFileName, clusterFileName, NClusters):
        clutoPath = os.getenv("CLUTO_BIN_PATH", ".")
        clutoExecutable = os.path.join(clutoPath, "vcluster")

        command = ([clutoExecutable]
                   + [matrixFileName, str(NClusters)]
                   + ["-clustfile=" + clusterFileName])

        # ["-clmethod=direct"])

        logger.info("Running CLUTO: %s", " ".join(command))
        clutoOutput = subprocess.check_output(command)
        logger.info("\n" + clutoOutput)

    def _computeClusters(self, sentenceVectors, NClusters):
        tmpDirName = tempfile.mkdtemp()
        matrixFileName = os.path.join(tmpDirName, "matrixFile")
        clusterFileName = os.path.join(tmpDirName, "clusterFile")

        logger.info("Saving matrix file: %s", matrixFileName)
        np.savetxt(matrixFileName,
                   sentenceVectors,
                   header=" ".join(map(str, sentenceVectors.shape)),
                   comments='')

        self._executeCLUTO(matrixFileName, clusterFileName, NClusters)

        self._sentenceIdClusters = [[] for _ in xrange(NClusters)]

        logger.info("Reading clusters file: %s", clusterFileName)
        with open(clusterFileName) as clusterFile:
            for i, line in enumerate(clusterFile):
                self._sentenceIdClusters[int(line)].append(i)

        logger.info("Removing temporary directory: %s", tmpDirName)
        shutil.rmtree(tmpDirName)

    def _compute(self, summarySentences):
        diversityReward = 0
        summarySentencesIds = map(lambda s: self._corpusSentenceMap[s],
                                  summarySentences)

        for sentenceCluster in self._sentenceIdClusters:
            diversityReward += math.sqrt(
                sum(map(lambda sI: self._singletonRewards[sI]
                        if sI in summarySentencesIds else 0,
                        sentenceCluster))
            )

        return diversityReward

    def setCorpus(self, corpus):
        logger.info("Preprocessing documents for diversity reward objective")
        self._corpus = corpus

        self._corpusSentenceList = corpus.getSentences()
        self._corpusSentenceVectos = corpus.getSentenceVectors()
        self._corpusLenght = len(self._corpusSentenceList)

        self._corpusSentenceMap = dict(
            zip(self._corpusSentenceList, range(self._corpusLenght))
        )

        self._similarities = sklearn.metrics.pairwise.cosine_similarity(
            self._corpusSentenceVectos
        )

        self._singletonRewards = self._similarities.mean(axis=1)

        self.K = int(math.ceil(self.kN * self._corpusLenght))

        logger.info("Clustering sentences")
        logger.info("Number of clusters: %d", self.K)
        self.clusters = self._computeClusters(
            self._corpusSentenceVectos,
            self.K
        )

    def getObjective(self, summary):
        def objective(sentence):
            return self._compute(summary.getSentences() + [sentence])

        return objective
