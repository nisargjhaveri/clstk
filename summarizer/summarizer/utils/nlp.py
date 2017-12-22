import os

import nltk

CORENLP_JAR = os.getenv("CORENLP_JAR")


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
