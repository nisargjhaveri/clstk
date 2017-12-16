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

    def setTranslation(self, translation):
        self._translation = translation

    def getTranslation(self):
        return self._translation

    def size(self):
        return len(self._text)
