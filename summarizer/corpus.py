import os

import nltk

from sentence import Sentence
from sentenceCollection import SentenceCollection


class Corpus(SentenceCollection):
    def __init__(self, dirname):
        super(Corpus, self).__init__()

        self._dirname = dirname

        self._prepareSentenceSplitter()

        self._documents = []

    def _prepareSentenceSplitter(self):
        self._sentenceSplitter = lambda doc: sum(
            map(lambda p: nltk.sent_tokenize(p, 'english'), doc.split("\n")),
            []
        )

    def load(self, params, translate=False):
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
        self.generateSentenceVectors(params['sourceLang'])

        if translate:
            self.translate(params['sourceLang'], params['targetLang'])
            self.generateTranslationSentenceVectors(params['sourceLang'])

        return self
