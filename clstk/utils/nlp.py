# -*- coding: utf-8 -*-

from collections import defaultdict

import nltk

# CORENLP_JAR = os.getenv("CORENLP_JAR")


def getSentenceSplitter():
    """
    Get sentence splitter function

    :returns: A function which takes a string and return list of sentence
              as strings.
    """
    def _sent_splitter(text):
        return nltk.sent_tokenize(text, 'english')

    return _sent_splitter


def getTokenizer(lang):
    """
    Get tokenizer for a given language

    :param lang: language
    :returns: tokenizer, which takes a sentence as string and returns list of
              tokens
    """
    # from nltk.tokenize.stanford import CoreNLPTokenizer
    # return CoreNLPTokenizer().tokenize

    # from polyglot.text import Text
    # return lambda t: Text(t).words

    # from nltk.tokenize.stanford import StanfordTokenizer
    # return StanfordTokenizer(CORENLP_JAR).tokenize

    if lang == 'en':
        return nltk.word_tokenize
    else:
        from polyglot.text import Text
        return lambda t: Text(t).words


def getDetokenizer(lang):
    """
    Get detokenizer for a given language

    :param lang: language
    :returns: detokenizer, which takes list of tokens and returns a sentence
              as string
    """
    d = nltk.tokenize.treebank.TreebankWordDetokenizer()
    return d.detokenize


def getStemmer():
    """
    Get stemmer. For now returns Porter Stemmer

    :returns: stemmer, which takes a token and returns its stem
    """
    return nltk.stem.PorterStemmer().stem


def getStopwords(lang):
    """
    Get list of stopwords for a given language

    :param lang: language
    :returns: list of stopwords including common puncuations
    """
    stopwords = defaultdict(list, {
        # 'gu': ["છે", u"અને", u"આ", u"પણ", u"કે", u"માટે", u"જ", u"એક", u"પર", u"હોય", u"જાય", u"તો", u"થઈ", u"થાય", u"આવે", u"વધારે", u"સાથે", u"કરી", u"નથી", u"જે", u"સુધી", u"શકે", u"પછી", u"કરે", u"અહીં", u"એ", u"કોઈ", u"તથા", u"રીતે", u"દૂર", u"કારણે", u"જો", u"કરવામાં", u"આવેલ", u"રહે", u"શકાય", u"રોગ", u"તમે", u"ન", u"અથવા", u"તેમજ", u"ખૂબ", u"તે", u"દ્વારા", u"સૌથી", u"આવી", u"પરંતુ", u"ઉપયોગ", u"મળે"],  # noqa: E501
        # 'hi': [u"है", u"के", u"में", u"से", u"की", u"का", u"हैं", u"को", u"और", u"पर", u"भी", u"हो", u"एक", u"लिए", u"यह", u"ही", u"इस", u"तो", u"जाता", u"नहीं", u"कि", u"होता", u"या", u"यहाँ", u"कर", u"तथा", u"व", u"तक", u"होती", u"होने", u"करने", u"जाती", u"जो", u"एवं", u"था", u"कारण", u"किया", u"ने", u"सकता", u"जा", u"कुछ", u"कम", u"साथ", u"न", u"चाहिए"],  # noqa: E501
        'en': nltk.corpus.stopwords.words('english'),
    })

    return (
        stopwords[lang]
        + ". , ; : ? ! ( ) [ ] \{ \}".split()
        + "/ \ | ~ @ # $ % ^ & * _ - + = ` `` ' '' \" < >".split()
    )
