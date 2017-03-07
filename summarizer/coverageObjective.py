import sklearn.metrics.pairwise

from objective import Objective


class CoverageObjective(Objective):
    def __init__(self, alpha):
        self.alpha = alpha

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
        self._corpus = corpus

        self._corpusSentenceList = corpus.getSentences()
        self._corpusLenght = len(self._corpusSentenceList)

        self._corpusSentenceMap = dict(
            zip(self._corpusSentenceList, range(self._corpusLenght))
        )

        self._similarities = sklearn.metrics.pairwise.cosine_similarity(
            corpus.getSentenceVectors()
        )

        self._corpusCoverage = map(
            lambda sI:
                self._computeIndividualCoverage(sI, self._corpusSentenceList),
            xrange(self._corpusLenght)
        )

    def getObjective(self, summary):
        def objective(sentence):
            return self._compute(summary.getSentences() + [sentence])

        return objective
