import os

import nltk
import sklearn

from sentence import Sentence
from sentenceCollection import SentenceCollection


class Corpus(SentenceCollection):
    def __init__(self, dirname):
        super(Corpus, self).__init__()

        self._dirname = dirname

        self._prepareSentenceSplitter()
        self._prepareTokenizer()
        self._prepareStemmer()
        self._prepareStopWordRemover()

        self._documents = []

    def _prepareSentenceSplitter(self):
        self._sentenceSplitter = nltk.data.load(
            'tokenizers/punkt/english.pickle'
        ).tokenize

    def _prepareTokenizer(self):
        self._wordTokenizer = nltk.word_tokenize

    def _prepareStemmer(self):
        self._stemmer = nltk.stem.PorterStemmer()

    def _prepareStopWordRemover(self):
        # TODO
        # stopwords = nltk.corpus.stopwords.words('english')
        self._stopWordRemover = lambda tokens: tokens

    def _generateSentenceVectors(self):
        def _processSentence(sentence):
            tokens = map(self._stemmer.stem,
                         self._stopWordRemover(
                             self._wordTokenizer(sentence.getLowerText())
                         ))

            return tokens

        sentenceVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
                                analyzer=_processSentence,
                            )

        sentenceVectors = sentenceVectorizer.fit_transform(
                                    self._sentences
                                )

        map(Sentence.setVector, self._sentences, sentenceVectors)

    def load(self):
        files = map(lambda f: os.path.join(self._dirname, f),
                    os.walk(self._dirname).next()[2])

        for filename in files:
            with open(filename) as f:
                document = f.read().decode('utf-8')

                self._documents.append(document)
                self.addSentences(map(Sentence,
                                      self._sentenceSplitter(document)))

        self._generateSentenceVectors()
