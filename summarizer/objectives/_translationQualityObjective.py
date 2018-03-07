import shelve

from ..utils.param import Param
from ..utils import nlp

from ._objective import Objective

from ..tqe import getPredictor

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
            self.sentenceScoresMap[sent] = 1 - self.sentenceScoresMap[sent]

    def setCorpus(self, corpus):
        logger.info("Processing documents for translation quality objective")
        self._corpus = corpus

        self._corpusSentenceList = corpus.getSentences()
        self._corpusLenght = len(self._corpusSentenceList)

        def _prepareSrcSentence(sentence):
            tokenize = nlp.getTokenizer()
            return " ".join(tokenize(sentence.getText()))

        def _prepareMtSentence(sentence):
            tokenize = nlp.getTokenizer()
            return " ".join(tokenize(sentence.getTranslation()))

        def getCacheKey(src, mt):
            return "_".join([src, mt]).encode('utf-8')

        srcSentences = map(_prepareSrcSentence, self._corpusSentenceList)
        mtSentences = map(_prepareMtSentence, self._corpusSentenceList)

        self.sentenceScoresMap = {}

        try:
            cache = shelve.open(self.cachePath, 'r')
            toPredict = []
            for sent, src, mt in zip(self._corpusSentenceList,
                                     srcSentences, mtSentences):
                if getCacheKey(src, mt) not in cache:
                    toPredict.append((sent, src, mt))
                else:
                    self.sentenceScoresMap[sent] = cache[getCacheKey(src, mt)]
            cache.close()
        except Exception:
            toPredict = zip(self._corpusSentenceList,
                            srcSentences, mtSentences)

        if len(toPredict):
            logger.info("Predicting translation quality of sentences")
            sentToPredict, srcToPredict, mtToPredict = zip(*toPredict)

            predictor = getPredictor(self.modelPath)
            predictedScores = predictor(srcToPredict, mtToPredict)

            writeCache = shelve.open(self.cachePath)
            for sent, src, mt, score in zip(sentToPredict,
                                            srcToPredict, mtToPredict,
                                            predictedScores):
                writeCache[getCacheKey(src, mt)] = score
                self.sentenceScoresMap[sent] = score
            writeCache.close()

        self._transformSentenceScores()

    def getObjective(self, summary):
        def objective(sentence):
            return self._compute(summary.getSentences() + [sentence])

        return objective
