from ..utils.param import Param

from ._objective import Objective

import logging
logger = logging.getLogger("translationQualityObjective.py")


class TranslationQualityObjective(Objective):
    def __init__(self, params):
        self.modelPath = params['model']
        self.cachePath = self.modelPath + '.cache'

    @staticmethod
    def getParams():
        return [
            Param(
                'model', type=str, default=None, metavar="path_to_model",
                help='Path to the trained Translation Quality Estimator model'
            )
        ]

    def _compute(self, summarySentences):
        return sum([self.sentenceScoresMap[s] for s in summarySentences])

    def _transformSentenceScores(self):
        for sent in self.sentenceScoresMap:
            self.sentenceScoresMap[sent] = (
                (1 - self.sentenceScoresMap[sent]) ** 4
            )

    def setCorpus(self, corpus):
        self._corpus = corpus

        self._corpusSentenceList = corpus.getSentences()
        self._corpusLenght = len(self._corpusSentenceList)

        from ..qualityEstimation.qualityEstimation import estimate
        estimate(corpus, self.modelPath)

        self.sentenceScoresMap = {}

        for sent in self._corpusSentenceList:
            self.sentenceScoresMap[sent] = sent.getExtra('qeScore')

        self._transformSentenceScores()

    def getObjective(self, summary):
        def objective(sentence):
            return self._compute(summary.getSentences() + [sentence])

        return objective
