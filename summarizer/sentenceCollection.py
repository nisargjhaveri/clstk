from sentence import Sentence
from translate.googleTranslate import translate
from simplify.neuralTextSimplification import simplify

import numpy as np
import sklearn

from utils import nlp


class SentenceCollection(object):
    def __init__(self):
        self._sentences = []

    def setSourceLang(self, lang):
        self.sourceLang = lang

    def setTargetLang(self, lang):
        self.targetLang = lang

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
                         nlp.getTokenizer(lang)(sentenceText.lower())
                         )

            return tokens

        sentenceVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
                                preprocessor=getText,
                                tokenizer=_tokenizeSentence,
                                stop_words=nlp.getStopwords(lang),
                                ngram_range=(1, 2)
                            )

        sentenceVectors = sentenceVectorizer.fit_transform(
                                    self._sentences
                                ).toarray()

        map(setVector, self._sentences, sentenceVectors)

    def generateSentenceVectors(self):
        self._generateSentenceVectors(self.sourceLang,
                                      Sentence.getText,
                                      Sentence.setVector)

    def generateTranslationSentenceVectors(self):
        self._generateSentenceVectors(self.targetLang,
                                      Sentence.getTranslation,
                                      Sentence.setTranslationVector)

    def translate(self, sourceLang, targetLang, replaceOriginal=False):
        text = "\n".join(map(Sentence.getText, self._sentences))
        translation, _ = translate(text, sourceLang, targetLang)

        translations = translation.split("\n")

        if replaceOriginal:
            map(Sentence.setText, self._sentences, translations)

        map(Sentence.setTranslation, self._sentences, translations)

    def simplify(self, sourceLang, replaceOriginal=False):
        sentences = map(Sentence.getText, self._sentences)

        simpleSentences = simplify(sentences, sourceLang)

        if replaceOriginal:
            map(Sentence.setText, self._sentences, simpleSentences)
            map(Sentence.setTranslation, self._sentences, simpleSentences)

        map(lambda s, t: s.setExtra('simpleText', t),
            self._sentences, simpleSentences)
