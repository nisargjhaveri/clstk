import os

import nltk

CORENLP_JAR = os.getenv("CORENLP_JAR")


def getSentenceSplitter():
    def _sent_splitter(text):
        return nltk.sent_tokenize(text, 'english')

    return _sent_splitter


def getTokenizer():
    # from nltk.tokenize.stanford import CoreNLPTokenizer
    # return CoreNLPTokenizer().tokenize

    # from polyglot.text import Text
    # return lambda t: Text(t).words

    # from nltk.tokenize.stanford import StanfordTokenizer
    # return StanfordTokenizer(CORENLP_JAR).tokenize

    return nltk.word_tokenize


def getStemmer():
    return nltk.stem.PorterStemmer().stem


def getStopwords():
    return (
        nltk.corpus.stopwords.words('english')
        + ". , ; : ? ! ( ) [ ] \{ \}".split()
        + "/ \ | ~ @ # $ % ^ & * _ - + = ` `` ' '' \" < >".split()
    )
