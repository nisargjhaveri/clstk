class Sentence(object):
    def __init__(self, sentenceText):
        self.setText(sentenceText)
        self.setTranslation(sentenceText)

        self._extras = {}

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

    def setExtra(self, key, value):
        self._extras[key] = value

    def getExtra(self, key, default=None):
        return self._extras[key] if key in self._extras else default

    def charCount(self):
        return len(self._translation)

    def tokenCount(self):
        return len(self._translation.split())
