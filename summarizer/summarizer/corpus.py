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
        self._prepareStopwords()

        self._documents = []

    def _prepareSentenceSplitter(self):
        self._sentenceSplitter = lambda doc: sum(
            map(lambda p: nltk.sent_tokenize(p, 'english'), doc.split("\n")),
            []
        )

    def _prepareTokenizer(self):
        self._wordTokenizer = nltk.word_tokenize

    def _prepareStemmer(self):
        self._stemmer = nltk.stem.PorterStemmer()

    def _prepareStopwords(self):
        self._stopwords = (
            nltk.corpus.stopwords.words('english')
            + ". , ; : ? ! ( ) [ ] \{ \}".split()
            + "/ \ | ~ @ # $ % ^ & * _ - + = ` `` ' '' \" < >".split()
        )

    def _generateSentenceVectors(self):
        def _tokenizeSentence(sentenceText):
            tokens = map(self._stemmer.stem,
                         self._wordTokenizer(sentenceText.lower())
                         )

            return tokens

        sentenceVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
                                preprocessor=Sentence.getText,
                                tokenizer=_tokenizeSentence,
                                stop_words=self._stopwords,
                                ngram_range=(1, 2)
                            )

        self._sentenceVectors = sentenceVectorizer.fit_transform(
                                    self._sentences
                                )

        map(Sentence.setVector, self._sentences, self._sentenceVectors)

    def getSentenceVectors(self):
        return self._sentenceVectors

    def load(self, params):
        # load corpus
        files = map(lambda f: os.path.join(self._dirname, f),
                    os.walk(self._dirname).next()[2])

        sentences = []

        for filename in files:
            with open(filename) as f:
                document = f.read().decode('utf-8')

                self._documents.append(document)
                sentences.extend(self._sentenceSplitter(document))

        sentences = map(lambda s: s.strip(), sentences)
        self.addSentences(map(Sentence, set(sentences)))

        # preprocess corpus
        self._generateSentenceVectors()

        return self
