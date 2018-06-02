"""
Translation Quality Estimation

Setup dependencies for TQE to use this. https://github.com/nisargjhaveri/tqe

You also need to train model using the said tqe system.
"""

import shelve
from ..utils import nlp

from tqe import getPredictor

import logging
logger = logging.getLogger("qualityEstimation.py")


def estimate(sentenceCollection, modelPath):
    """
    Estimate translation quality for each sentence in collection.
    It sets an extra value with key ``qeScore`` on each sentence.

    :param sentenceCollection: SentenceCollection to estimate quality
    :type sentenceCollection:
        :class:`clstk.sentenceCollection.SentenceCollection`

    .. seealso:: :meth:`clstk.sentence.Sentence.getExtra`
    """
    logger.info("Predicting translation quality of sentences")

    sourceLang = sentenceCollection.sourceLang
    targetLang = sentenceCollection.targetLang

    cachePath = modelPath + '.cache'

    def _prepareSrcSentence(sentence):
        tokenize = nlp.getTokenizer(sourceLang)
        return " ".join(tokenize(sentence.getText()))

    def _prepareMtSentence(sentence):
        tokenize = nlp.getTokenizer(targetLang)
        return " ".join(tokenize(sentence.getTranslation()))

    def getCacheKey(src, mt):
        return "_".join([src, mt]).encode('utf-8')

    _sentenceList = sentenceCollection.getSentences()

    srcSentences = map(_prepareSrcSentence, _sentenceList)
    mtSentences = map(_prepareMtSentence, _sentenceList)

    try:
        cache = shelve.open(cachePath, 'r')
        toPredict = []
        for sent, src, mt in zip(_sentenceList,
                                 srcSentences, mtSentences):
            if getCacheKey(src, mt) not in cache:
                toPredict.append((sent, src, mt))
            else:
                sent.setExtra('qeScore', cache[getCacheKey(src, mt)])
        cache.close()
    except Exception:
        toPredict = zip(_sentenceList,
                        srcSentences, mtSentences)

    if len(toPredict):
        sentToPredict, srcToPredict, mtToPredict = zip(*toPredict)

        predictor = getPredictor(modelPath)
        predictedScores = predictor(srcToPredict, mtToPredict)

        writeCache = shelve.open(cachePath)
        for sent, src, mt, score in zip(sentToPredict,
                                        srcToPredict, mtToPredict,
                                        predictedScores):
            writeCache[getCacheKey(src, mt)] = score
            sent.setExtra('qeScore', score)
        writeCache.close()
