from sentence import Sentence
from translate.googleTranslate import translate
from simplify.neuralTextSimplification import simplify

import numpy as np
import sklearn

from utils import nlp


class SentenceCollection(object):
    """
    Class to store a colelction of sentences.

    Also proivdes several common operations on the collection.
    """
    def __init__(self):
        """
        Initialize the collection
        """
        self._sentences = []

    def setSourceLang(self, lang):
        """
        Set source language for the colelction

        :param lang: two-letter code for source language
        """
        self.sourceLang = lang

    def setTargetLang(self, lang):
        """
        Set target language for the colelction

        :param lang: two-letter code for target language
        """
        self.targetLang = lang

    def addSentence(self, sentence):
        """
        Add a sentence to the colelction

        :param sentence: sentence to be added
        """
        if not isinstance(sentence, Sentence):
            raise RuntimeError("Expected an object of Sentence class")

        self._sentences.append(sentence)

    def addSentences(self, sentences):
        """
        Add sentences to the colelction

        :param sentences: list of sentence to be added

        .. seealso::
            :meth:`clstk.sentenceCollection.SentenceCollection.addSentence`
        """
        map(self.addSentence, sentences)

    def getSentences(self):
        """
        Get list of sentences in the collection

        :returns: list of sentences
        """
        return self._sentences[:]

    def getSentenceVectors(self):
        """
        Get list of sentence vectors for sentences in the collection

        :returns: :class:`np.array` containing sentence vectors
        """
        return np.array(map(Sentence.getVector, self._sentences))

    def getTranslationSentenceVectors(self):
        """
        Get list of sentence vectors for translations of sentences in the
        collection

        :returns: :class:`np.array` containing sentence vectors
        """
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
        """
        Generate sentence vectors
        """
        self._generateSentenceVectors(self.sourceLang,
                                      Sentence.getText,
                                      Sentence.setVector)

    def generateTranslationSentenceVectors(self):
        """
        Generate sentence vectors for translations
        """
        self._generateSentenceVectors(self.targetLang,
                                      Sentence.getTranslation,
                                      Sentence.setTranslationVector)

    def translate(self, sourceLang, targetLang, replaceOriginal=False):
        """
        Translate sentences

        :param sourceLang: two-letter code for source language
        :param targetLang: two-letter code for target language
        :param replaceOriginal: Replace source text with translation if
                                ``True``. Used for early-translation
        """
        text = "\n".join(map(Sentence.getText, self._sentences))
        translation, _ = translate(text, sourceLang, targetLang)

        translations = translation.split("\n")

        if replaceOriginal:
            map(Sentence.setText, self._sentences, translations)

        map(Sentence.setTranslation, self._sentences, translations)

    def simplify(self, sourceLang, replaceOriginal=False):
        """
        Simplify sentences

        :param sourceLang: two-letter code for language
        :param replaceOriginal: Replace source sentences with simplified
                                sentences. Used for early-simplify.
        """
        sentences = map(Sentence.getText, self._sentences)

        simpleSentences = simplify(sentences, sourceLang)

        if replaceOriginal:
            map(Sentence.setText, self._sentences, simpleSentences)
            map(Sentence.setTranslation, self._sentences, simpleSentences)

        map(lambda s, t: s.setExtra('simpleText', t),
            self._sentences, simpleSentences)
