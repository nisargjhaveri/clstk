import os
import subprocess
import collections
import cPickle
import regex

import numpy as np
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from nltk.parse import CoreNLPParser

from multiprocessing.dummy import Pool

import kenlm

import logging
logger = logging.getLogger("root")


def _loadSentences(filePath):
    with open(filePath) as lines:
        sentences = map(
            lambda s: s.decode('utf-8').strip(), list(lines))

    return np.array(sentences, dtype=object)


def _normalizeSentences(sentences):
    def _normalizeSentence(sentence):
        return sentence.lower()

    return map(_normalizeSentence, sentences)


def _trainLM(sentences, lmFilePath, order):
    kenlmBin = os.getenv("KENLM_BIN", None)
    if kenlmBin is None:
        raise RuntimeError("Environment variable KENLM_BIN is not set")

    kenlmExecutable = os.path.join(kenlmBin, "lmplz")

    with open(lmFilePath, "wb") as lmFile:
        devNull = None if logger.isEnabledFor(logging.INFO) \
                                            else open(os.devnull, 'w')
        lmplz = subprocess.Popen([kenlmExecutable, "-o", str(order)],
                                 stdin=subprocess.PIPE,
                                 stdout=lmFile,
                                 stderr=devNull)
        lmplz.communicate("\n".join(_normalizeSentences(sentences))
                          .encode("utf-8"))
        devNull is None or devNull.close()


def _getNGrams(sentence, n):
    tokens = sentence.split()
    count = len(tokens)

    return zip(*[tokens[i:count - n + i] for i in xrange(n)])


def _loadNGramCounts(sentences, ngramPath, trainNGrams=True):
    sentences = _normalizeSentences(sentences)

    ngrams1counter, ngrams2counter, ngrams3counter = None, None, None

    if (trainNGrams):
        logger.info("Computing ngram frequencies")
        ngrams1 = sum([_getNGrams(s, 1) for s in sentences], [])
        ngrams1counter = collections.Counter(ngrams1)

        ngrams2 = sum([_getNGrams(s, 2) for s in sentences], [])
        ngrams2counter = collections.Counter(ngrams2)

        ngrams3 = sum([_getNGrams(s, 3) for s in sentences], [])
        ngrams3counter = collections.Counter(ngrams3)

        ngramCounts = (ngrams1counter, ngrams2counter, ngrams3counter)

        with open(ngramPath, "wb") as ngramsFile:
            cPickle.dump(ngramCounts, ngramsFile, cPickle.HIGHEST_PROTOCOL)
    else:
        logger.info("Loading ngram frequencies")
        with open(ngramPath) as ngramsFile:
            ngramCounts = cPickle.load(ngramsFile)

    return ngramCounts


def _getHighLowFreqNGrams(counter):
    ngrams = map(lambda x: x[0], counter.most_common())
    totalCount = len(ngrams)
    highFreqNGrams = ngrams[:(totalCount / 4)]
    lowFreqNGrams = ngrams[(3 * totalCount / 4):]

    return set(highFreqNGrams), set(lowFreqNGrams)


def _getOverlapCount(sentence, ngrams, n):
    sentenceNGrams = _getNGrams(sentence, n)

    count = 0
    for ngram in sentenceNGrams:
        if ngram in ngrams:
            count += 1

    return count


def _getParseTrees(srcSentences, parsePath, parseSentences):
    if parseSentences:
        logger.info("Parsing sentences")
        p = Pool(10)

        parser = CoreNLPParser(
                    url=os.getenv("CORENLP_HOST", "http://localhost:9000"))
        parses = p.map(lambda s: parser.parse_one(s.split()), srcSentences)

        with open(parsePath, "wb") as ngramsFile:
            cPickle.dump(parses, ngramsFile, cPickle.HIGHEST_PROTOCOL)
    else:
        logger.info("Loading parse trees")
        with open(parsePath) as ngramsFile:
            parses = cPickle.load(ngramsFile)

    return parses


def _getFeatures(srcSentences, mtSentences, refSentences, train_index,
                 fileBasename, parseFileSuffix,
                 trainLM=True, trainNGrams=True, parseSentences=True):
    srcLMPath = fileBasename + ".src.lm.2.arpa"
    refLMPath = fileBasename + ".ref.lm.2.arpa"
    ngramPath = fileBasename + ".src.ngrams.pickle"
    parsePath = fileBasename + ".src" + parseFileSuffix + ".parse"

    if trainLM:
        logger.info("Training language models")
        _trainLM(srcSentences[train_index], srcLMPath, 2)
        _trainLM(refSentences[train_index], refLMPath, 2)

    logger.info("Loading language models")
    srcModel = kenlm.Model(srcLMPath)
    refModel = kenlm.Model(refLMPath)

    ngramCounts = _loadNGramCounts(srcSentences[train_index], ngramPath,
                                   trainNGrams)
    high1grams, low1grams = _getHighLowFreqNGrams(ngramCounts[0])
    high2grams, low2grams = _getHighLowFreqNGrams(ngramCounts[1])
    high3grams, low3grams = _getHighLowFreqNGrams(ngramCounts[2])

    srcParses = _getParseTrees(srcSentences, parsePath, parseSentences)

    posCounts = CountVectorizer(
        lowercase=False,
        tokenizer=lambda t: map(lambda p: p[1], t.pos()),
        ngram_range=(1, 2)
    )
    posCounts.fit(srcParses)

    punc = regex.compile(r'[^\w]', regex.UNICODE)

    def _computeSentenceFeatures(srcSentence, mtSentence, srcParse):
        srcTokens = srcSentence.split()
        mtTokens = mtSentence.split()

        srcCount = float(len(srcTokens))

        features = [
            len(srcTokens),
            len(mtTokens),
            np.mean(map(len, srcTokens)),
            np.mean(map(len, mtTokens)),
            float(len(srcTokens)) / float(len(mtTokens)),
            float(len(mtTokens)) / float(len(srcTokens)),
            len(filter(punc.search, srcTokens)),
            len(filter(punc.search, mtTokens)),
            float(len(set(mtTokens))) / float(len(mtTokens)),
            srcModel.score(srcSentence),
            refModel.score(mtSentence),
            _getOverlapCount(srcSentence, high1grams, 1) / srcCount,
            _getOverlapCount(srcSentence, low1grams, 1) / srcCount,
            _getOverlapCount(srcSentence, high2grams, 2) / srcCount,
            _getOverlapCount(srcSentence, low2grams, 2) / srcCount,
            _getOverlapCount(srcSentence, high3grams, 3) / srcCount,
            _getOverlapCount(srcSentence, low3grams, 3) / srcCount,
            srcParse.height(),
        ]

        features.extend(posCounts.transform([srcParse]).todense().tolist()[0])

        return features

    logger.info("Computing features")
    X = np.array(
                map(_computeSentenceFeatures,
                    srcSentences,
                    mtSentences,
                    srcParses)
                )

    X = preprocessing.normalize(X)

    pca = PCA(n_components=15)
    X = pca.fit_transform(X)

    return X


def plotData(X, y, svr):
    pca = PCA(n_components=2)
    pcaX = pca.fit_transform(X)

    # X_plot = np.linspace(0, 5, 100000)[:, None]
    # y_svr = svr.predict(X_plot)

    # sv_ind = svr.support_
    # plt.scatter(pcaX[sv_ind], y[sv_ind], c='r', label='SVR support vectors',
    #             zorder=2)
    plt.scatter(pcaX[:, 0], pcaX[:, 1], c=y, cmap=cm.Oranges, label='data')
    # plt.plot(X_plot, y_svr, c='r',  label='SVR')

    # plt.xlabel('data')
    # plt.ylabel('target')
    # plt.title('')
    plt.legend()

    plt.show()


def train_model(workspaceDir, modelName, evaluate=False, evalFileSuffix=None,
                featureFileSuffix=None,
                trainLM=True, trainNGrams=True, parseSentences=True):
    logger.info("initializing TQE training")
    fileBasename = os.path.join(workspaceDir, "tqe." + modelName)

    targetPath = fileBasename + ".hter"
    srcSentencesPath = fileBasename + ".src"
    mtSentencesPath = fileBasename + ".mt"
    refSentencesPath = fileBasename + ".ref"

    srcSentences = _loadSentences(srcSentencesPath)
    mtSentences = _loadSentences(mtSentencesPath)
    refSentences = _loadSentences(refSentencesPath)

    if evaluate and not evalFileSuffix:
        logger.info("Creating train test split")
        splitter = ShuffleSplit(n_splits=1, test_size=.1, random_state=42)
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=0, random_state=42)

    for train_index, test_index in splitter.split(srcSentences):
        y = np.clip(np.loadtxt(targetPath), 0, 1)
        X = np.loadtxt(fileBasename + featureFileSuffix) \
            if featureFileSuffix \
            else _getFeatures(srcSentences, mtSentences, refSentences,
                              train_index,
                              fileBasename, "",
                              trainLM=trainLM, trainNGrams=trainNGrams,
                              parseSentences=parseSentences)

        X_train = X[train_index]
        y_train = y[train_index]

        logger.info("Training SVR")
        svr = svm.SVR(verbose=True)
        svr.fit(X_train, y_train)

        # plotData(X_train, y_train, svr)

        if evaluate:
            logger.info("Evaluating")
            if evalFileSuffix:
                X_test = np.loadtxt(fileBasename + featureFileSuffix +
                                    evalFileSuffix) \
                    if featureFileSuffix \
                    else _getFeatures(
                            _loadSentences(srcSentencesPath + evalFileSuffix),
                            _loadSentences(mtSentencesPath + evalFileSuffix),
                            None, [], fileBasename, evalFileSuffix,
                            trainLM=False, trainNGrams=False,
                            parseSentences=parseSentences
                            )
                y_test = np.clip(np.loadtxt(targetPath + evalFileSuffix), 0, 1)
            else:
                X_test = X[test_index]
                y_test = y[test_index]

            y_pred = svr.predict(X_test)
            _evaluate(y_pred, y_test)


def _evaluate(y_pred, y_test):
    print "MSE:", mean_squared_error(y_test, y_pred)
    print "MAE:", mean_absolute_error(y_test, y_pred)
    print "Pearson's r:", pearsonr(y_pred, y_test)
