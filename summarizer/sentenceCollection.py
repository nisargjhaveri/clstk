from sentence import Sentence
from translate.googleTranslate import translate

import numpy as np


class SentenceCollection(object):
    def __init__(self):
        self._sentences = []

    def addSentence(self, sentence):
        if not isinstance(sentence, Sentence):
            raise RuntimeError("Expected an object of Sentence class")

        self._sentences.append(sentence)

    def addSentences(self, sentences):
        map(self.addSentence, sentences)

    def getSentences(self):
        return self._sentences[:]

    def getSentenceVectors(self):
        return np.array(map(Sentence.getVector, self._sentences))

    def translateCollection(self, sourceLang, targetLang):
        text = "\n".join(map(Sentence.getText, self._sentences))
        translation, _ = translate(text, sourceLang, targetLang)

        map(Sentence.setTranslation, self._sentences, translation.split("\n"))
