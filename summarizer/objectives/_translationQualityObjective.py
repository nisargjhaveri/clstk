from ..utils.param import Param
from ..utils import nlp

from ._objective import Objective

from ..tqe import getPredictor

import logging
logger = logging.getLogger("translationQualityObjective.py")


class TranslationQualityObjective(Objective):
    def __init__(self, params):
        self.modelPath = params['model']

    @staticmethod
    def getParams():
        return [
            Param(
                'model', type=str, default=None, metavar="path_to_model",
                help='Path to the trained Translation Quality Estimator model'
            )
        ]

    def _compute(self, summarySentences):
        summarySentenceIds = map(lambda s: self._corpusSentenceMap[s],
                                 summarySentences)

        return sum([self.sentenceScores[id] for id in summarySentenceIds])

    def setCorpus(self, corpus):
        logger.info("Processing documents for translation quality objective")
        self._corpus = corpus

        self._corpusSentenceList = corpus.getSentences()
        self._corpusLenght = len(self._corpusSentenceList)

        self._corpusSentenceMap = dict(
            zip(self._corpusSentenceList, range(self._corpusLenght))
        )

        predictor = getPredictor(self.modelPath)

        def _prepareSrcSentence(sentence):
            tokenize = nlp.getTokenizer()
            return " ".join(tokenize(sentence.getText()))

        def _prepareMtSentence(sentence):
            tokenize = nlp.getTokenizer()
            return " ".join(tokenize(sentence.getTranslation()))

        srcSentences = map(_prepareSrcSentence, self._corpusSentenceList)
        mtSentences = map(_prepareMtSentence, self._corpusSentenceList)

        logger.info("Predicting translation quality of sentences")
        self.sentenceScores = predictor(srcSentences, mtSentences)

        self.sentenceScores = map(lambda x: 1 - x, self.sentenceScores)

        print self.sentenceScores

    def getObjective(self, summary):
        def objective(sentence):
            return self._compute(summary.getSentences() + [sentence])

        return objective
