import sklearn.metrics.pairwise

from ..utils.param import Param
from ._objective import Objective

import logging
logger = logging.getLogger("coverageObjective.py")


class CoverageObjective(Objective):
    def __init__(self, params):
        self.alphaN = params['alpha']

    @staticmethod
    def getParams():
        return [
            Param(
                'alpha', type=float, default=6.0, metavar="alphaN",
                help='Threshold co-efficient to be used in coverage objective.'
                + ' The co-efficient the  will be calucated as alphaN / N'
            )
        ]

    def _computeIndividualCoverage(self, sentenceIndex, sentenceList):
        return sum(
            map(
                lambda s:
                    self._similarities[sentenceIndex][
                        self._corpusSentenceMap[s]
                    ],
                sentenceList
            )
        )

    def _compute(self, summarySentences):
        coverage = 0
        for sentenceIndex in xrange(self._corpusLenght):
            coverage += min(
                self._computeIndividualCoverage(sentenceIndex,
                                                summarySentences),
                self.alpha * self._corpusCoverage[sentenceIndex]
            )

        return coverage

    def setCorpus(self, corpus):
        logger.info("Preprocessing documents for coverage objective")
        self._corpus = corpus

        self._corpusSentenceList = corpus.getSentences()
        self._corpusLenght = len(self._corpusSentenceList)

        self._corpusSentenceMap = dict(
            zip(self._corpusSentenceList, range(self._corpusLenght))
        )

        self._similarities = sklearn.metrics.pairwise.cosine_similarity(
            corpus.getSentenceVectors()
        )
        # + 0.5 * sklearn.metrics.pairwise.cosine_similarity(
        #     corpus.getTranslationSentenceVectors()
        # )

        self._corpusCoverage = map(
            lambda sI:
                self._computeIndividualCoverage(sI, self._corpusSentenceList),
            xrange(self._corpusLenght)
        )

        self.alpha = float(self.alphaN) / self._corpusLenght if (
                            self.alphaN is not None
                        ) else 1

    def getObjective(self, summary):
        def objective(sentence):
            return self._compute(summary.getSentences() + [sentence])

        return objective
