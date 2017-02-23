class Sentence(object):
    def __init__(self, sentenceText):
        global id
        self.setText(sentenceText)

    def setText(self, sentenceText):
        self._text = sentenceText.strip()

    def getText(self):
        return self._text

    def setVector(self, vector):
        self._vector = vector

    def getVector(self):
        return self._vector

    def size(self):
        return len(self._text)
