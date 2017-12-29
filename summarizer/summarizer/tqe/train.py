import os
import subprocess
import collections
import cPickle
# import regex

import numpy as np
from sklearn import svm
from sklearn.model_selection import ShuffleSplit

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# import kenlm

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


def _getFeatures(srcSentences, mtSentences, refSentences, train_index,
                 fileBasename, trainLM=True, trainNGrams=True):
    srcLMPath = fileBasename + ".src.lm.2.arpa"
    refLMPath = fileBasename + ".ref.lm.2.arpa"
    ngramPath = fileBasename + ".src.ngrams.pickle"

    if trainLM:
        logger.info("Training language models")
        _trainLM(srcSentences[train_index], srcLMPath, 2)
        _trainLM(refSentences[train_index], refLMPath, 2)

    # logger.info("Loading language models")
    # srcModel = kenlm.Model(srcLMPath)
    # refModel = kenlm.Model(refLMPath)

    ngramCounts = _loadNGramCounts(srcSentences[train_index], ngramPath,
                                   trainNGrams)
    high1grams, low1grams = _getHighLowFreqNGrams(ngramCounts[0])
    high2grams, low2grams = _getHighLowFreqNGrams(ngramCounts[1])
    high3grams, low3grams = _getHighLowFreqNGrams(ngramCounts[2])

    def _computeSentenceFeatures(srcSentence, mtSentence):
        srcTokens = srcSentence.split()
        mtTokens = mtSentence.split()

        srcCount = float(len(srcTokens))

        # punc = regex.compile(r'[^\w]', regex.UNICODE)

        return [
            len(srcTokens),
            # len(mtTokens),
            np.mean(map(len, srcTokens)),
            np.mean(map(len, mtTokens)),
            # float(len(srcTokens)) / float(len(mtTokens)),
            # float(len(mtTokens)) / float(len(srcTokens)),
            # len(filter(punc.search, srcTokens)),
            # len(filter(punc.search, mtTokens)),
            float(len(set(mtTokens))) / float(len(mtTokens)),
            # srcModel.score(srcSentence),
            # refModel.score(mtSentence),
            _getOverlapCount(srcSentence, high1grams, 1) / srcCount,
            # _getOverlapCount(srcSentence, low1grams, 1) / srcCount,
            # _getOverlapCount(srcSentence, high2grams, 2) / srcCount,
            # _getOverlapCount(srcSentence, low2grams, 2) / srcCount,
            # _getOverlapCount(srcSentence, high3grams, 3) / srcCount,
            # _getOverlapCount(srcSentence, low3grams, 3) / srcCount,
        ]

    logger.info("Computing features")
    return np.array(
                map(_computeSentenceFeatures,
                    srcSentences,
                    mtSentences)
                )


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


def train_model(workspaceDir, modelName, evaluate=False,
                trainLM=True, trainNGrams=True):
    logger.info("initializing TQE training")
    fileBasename = os.path.join(workspaceDir, "tqe." + modelName)

    targetPath = fileBasename + ".hter"
    srcSentencesPath = fileBasename + ".src"
    mtSentencesPath = fileBasename + ".mt"
    refSentencesPath = fileBasename + ".ref"

    srcSentences = _loadSentences(srcSentencesPath)
    mtSentences = _loadSentences(mtSentencesPath)
    refSentences = _loadSentences(refSentencesPath)

    if evaluate:
        splitter = ShuffleSplit(n_splits=1, test_size=.1, random_state=42)
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=0, random_state=42)

    for train_index, test_index in splitter.split(srcSentences):
        y = np.clip(np.loadtxt(targetPath), 0, 1)
        X = _getFeatures(srcSentences, mtSentences, refSentences, train_index,
                         fileBasename, trainLM=trainLM,
                         trainNGrams=trainNGrams)

        X_train = X[train_index]
        y_train = y[train_index]

        logger.info("Training SVR")
        svr = svm.SVR(verbose=True)
        svr.fit(X_train, y_train)

        # plotData(X_train, y_train, svr)

        if evaluate:
            logger.info("Evaluating")
            X_test = X[test_index]
            y_test = y[test_index]

            y_pred = svr.predict(X_test)
            _evaluate(y_pred, y_test)


def _evaluate(y_pred, y_test):
    print "MSE:", mean_squared_error(y_test, y_pred)
    print "MAE:", mean_absolute_error(y_test, y_pred)
    print "Pearson's r:", pearsonr(y_pred, y_test)
