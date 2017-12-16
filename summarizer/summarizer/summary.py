from sentenceCollection import SentenceCollection


class Summary(SentenceCollection):
    def size(self):
        return sum(map(lambda s: s.size(), self._sentences))

    def getSummary(self):
        return "\n".join(map(lambda s: s.getText(), self._sentences))

    def getTargetSummary(self):
        return "\n".join(map(lambda s: s.getTranslation(), self._sentences))
