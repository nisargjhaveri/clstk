class Sentence(object):
    """
    Class to represent a single sentence
    """

    def __init__(self, sentenceText):
        """
        Set sentence text and translated text

        :param sentenceText: sentence text
        """
        self.setText(sentenceText)
        self.setTranslation(sentenceText)

        self._extras = {}

    def setText(self, sentenceText):
        """
        Set text for the sentence

        :param sentenceText: sentence text
        """
        self._text = sentenceText.strip()

    def getText(self):
        """
        Get sentence text

        :returns: sentence text
        """
        return self._text

    def setTranslation(self, translation):
        """
        Set translated text

        :param translation: translated text
        """
        self._translation = translation

    def getTranslation(self):
        """
        Get translated text

        The translated text defaults to sentence text

        :returns: translated text
        """
        # return " ".join(self._translationTokens)
        return self._translation

    def setVector(self, vector):
        """
        Set sentence vector

        :param vector: sentence vector
        """
        self._vector = vector

    def getVector(self):
        """
        Get sentence vector

        :returns: sentence vector
        """
        return self._vector

    def setTranslationVector(self, vector):
        """
        Set sentence vector for translated text

        :param vector: sentence vector
        """
        self._translationVector = vector

    def getTranslationVector(self):
        """
        Get sentence vector for translated text

        :returns: sentence vector
        """
        return self._translationVector

    def setExtra(self, key, value):
        """
        Set extra key-value pair

        :param key: key for the stored value
        :param value: value to store
        """
        self._extras[key] = value

    def getExtra(self, key, default=None):
        """
        Get extra value from key

        :param key: key for the stored value
        :param default: default value if key not found
        """
        return self._extras[key] if key in self._extras else default

    def charCount(self):
        """
        Get character count for translated text

        :returns: Number of character in translated text
        """
        return len(self._translation)

    def tokenCount(self):
        """
        Get token count for translated text

        :returns: Number of tokens in translated text
        """
        return len(self._translation.split())
