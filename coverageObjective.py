import sklearn.metrics.pairwise

from objective import Objective


class CoverageObjective(Objective):
    def __init__(self, alpha):
        self.alpha = alpha

    def _sentenceSimilarity(sentence1, sentence2):
        return sklearn.metrics.pairwise.cosine_similarity(
            [sentence1.getVector()],
            [sentence2.getVector()]
        )[0][0]

    def _computeIndividualCoverage(self, sentence, sentenceList):
        return sum(
            map(
                lambda s: self._sentenceSimilarity(sentence, s), sentenceList
            )
        )

    def _compute(self, summarySentences, corpusSentences):
        coverage = 0
        for sentence in corpusSentences:
            coverage += min(
                self._computeIndividualCoverage(sentence, summarySentences),
                self.alpha * self._computeIndividualCoverage(
                    sentence, corpusSentences
                )
            )

        return coverage

    def getObjective(self, summary, corpus):
        def objective(sentence):
            return self.compute(
                summary.getSentences() + sentence, corpus.getSentences()
            )
