from sentenceCollection import SentenceCollection


class Summary(SentenceCollection):
    def charCount(self):
        """
        Get total number of character in all the sentences
        """
        return sum(map(lambda s: s.charCount(), self._sentences))

    def tokenCount(self):
        """
        Get total number of tokens in all the sentences
        """
        return sum(map(lambda s: s.tokenCount(), self._sentences))

    def getSummary(self):
        """
        Get printable summary generated from source text
        """
        return "\n".join(map(lambda s: s.getText(), self._sentences))

    def getTargetSummary(self):
        """
        Get printable summary generated from translated text
        """
        return "\n".join(map(lambda s: s.getTranslation(), self._sentences))
