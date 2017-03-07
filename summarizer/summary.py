from sentenceCollection import SentenceCollection


class Summary(SentenceCollection):
    def size(self):
        return sum(map(lambda s: s.size(), self._sentences))

    def getFormattedSummary(self):
        return "\n".join(map(lambda s: s.getText(), self._sentences))
