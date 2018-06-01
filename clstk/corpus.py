import os

from sentence import Sentence
from sentenceCollection import SentenceCollection

from utils import nlp

import logging
logger = logging.getLogger("corpus.py")


class Corpus(SentenceCollection):
    """
    Class for source documents. Contains utilities for loading document set.
    """
    def __init__(self, dirname):
        """
        Initialize the class

        :param dirname: Directory from where source documents are to be loaded
        """
        super(Corpus, self).__init__()

        self._dirname = dirname

        self._prepareSentenceSplitter()

        self._documents = []

    def _prepareSentenceSplitter(self):
        self._sentenceSplitter = lambda doc: sum(
            map(lambda p: nlp.getSentenceSplitter()(p), doc.split("\n")),
            []
        )

    def load(self, params, translate=False, replaceWithTranslation=False,
             simplify=False, replaceWithSimplified=False):
        """
        Load source docuement set

        :param params: ``dict`` containing different params including
                       ``sourceLang`` and ``targetLang``.
        :param translate: Whether to translate sentences to target language
        :param replaceWithTranslation: Whether to replace source sentences
                                       with translation
        :param simplify: Whether to simplify sentences
        :param replaceWithSimplified: Whether to replace source sentences with
                                      simplified sentences
        """
        self.setSourceLang(params['sourceLang'])
        self.setTargetLang(params['targetLang'])

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

        if simplify:
            logger.info("Simplifying sentences")
            self.simplify(self.sourceLang,
                          replaceOriginal=replaceWithSimplified)

        if translate:
            if self.sourceLang != self.targetLang:
                logger.info("Translating sentences")
                self.translate(self.sourceLang,
                               self.targetLang,
                               replaceOriginal=replaceWithTranslation)

            if replaceWithTranslation:
                self.setSourceLang(self.targetLang)

            self.generateTranslationSentenceVectors()

        self.generateSentenceVectors()

        return self
