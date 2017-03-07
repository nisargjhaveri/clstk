from sentence import Sentence


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
