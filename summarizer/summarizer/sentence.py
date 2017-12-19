import nltk


class Sentence(object):
    def __init__(self, sentenceText):
        self.setText(sentenceText)
        self.setTranslation(sentenceText)

    def setText(self, sentenceText):
        self._text = sentenceText.strip()

    def getText(self):
        return self._text

    def getTokens(self):
        return self._tokens

    def setVector(self, vector):
        self._vector = vector

    def getVector(self):
        return self._vector

    def setTranslation(self, translation):
        self._translation = translation

        self._translationTokens = nltk.word_tokenize(self._text)

    def getTranslation(self):
        return self._translation

    def charCount(self):
        return len(self._text)

    def tokenCount(self):
        return len(self._translationTokens)
