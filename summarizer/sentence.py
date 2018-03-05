from utils import nlp

tokenize = nlp.getTokenizer()


class Sentence(object):
    def __init__(self, sentenceText):
        self.setText(sentenceText)
        self.setTranslation(sentenceText)

    def setText(self, sentenceText):
        self._text = sentenceText.strip()

    def getText(self):
        return self._text

    def setVector(self, vector):
        self._vector = vector

    def getVector(self):
        return self._vector

    def setTranslationVector(self, vector):
        self._translationVector = vector

    def getTranslationVector(self):
        return self._translationVector

    def setTranslation(self, translation):
        self._translation = translation

    def getTranslation(self):
        # return " ".join(self._translationTokens)
        return self._translation

    def charCount(self):
        return len(self._translation)

    def tokenCount(self):
        return len(self._translation.split())
