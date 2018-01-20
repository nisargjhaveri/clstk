from sentenceCollection import SentenceCollection


class Summary(SentenceCollection):
    def charCount(self):
        return sum(map(lambda s: s.charCount(), self._sentences))

    def tokenCount(self):
        return sum(map(lambda s: s.tokenCount(), self._sentences))

    def getSummary(self):
        return "\n".join(map(lambda s: s.getText(), self._sentences))

    def getTargetSummary(self):
        return "\n".join(map(lambda s: s.getTranslation(), self._sentences))
