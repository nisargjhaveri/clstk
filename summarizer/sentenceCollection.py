from sentence import Sentence
from translate.googleTranslate import translate

import numpy as np
import sklearn

from utils import nlp


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

    def getTranslationSentenceVectors(self):
        return np.array(map(Sentence.getTranslationVector, self._sentences))

    def _generateSentenceVectors(self, lang, getText, setVector):
        def _tokenizeSentence(sentenceText):
            tokens = map(nlp.getStemmer(),
                         nlp.getTokenizer()(sentenceText.lower())
                         )

            return tokens

        sentenceVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
                                preprocessor=getText,
                                tokenizer=_tokenizeSentence,
                                stop_words=nlp.getStopwords(),
                                ngram_range=(1, 2)
                            )

        sentenceVectors = sentenceVectorizer.fit_transform(
                                    self._sentences
                                ).toarray()

        map(setVector, self._sentences, sentenceVectors)

    def generateSentenceVectors(self, sourceLang):
        self._generateSentenceVectors(sourceLang,
                                      Sentence.getText,
                                      Sentence.setVector)

    def generateTranslationSentenceVectors(self, targetLang):
        self._generateSentenceVectors(targetLang,
                                      Sentence.getTranslation,
                                      Sentence.setTranslationVector)

    def translate(self, sourceLang, targetLang, replaceOriginal=False):
        text = "\n".join(map(Sentence.getText, self._sentences))
        translation, _ = translate(text, sourceLang, targetLang)

        translations = translation.split("\n")

        if replaceOriginal:
            map(Sentence.setText, self._sentences, translations)

        map(Sentence.setTranslation, self._sentences, translations)
