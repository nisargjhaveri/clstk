import os
import subprocess
import collections
import cPickle

import numpy as np
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

# from sklearn.feature_extraction.text import CountVectorizer

# from sklearn.decomposition import PCA

# import matplotlib.pyplot as plt
# from matplotlib import cm

from . import utils

from nltk.parse import CoreNLPParser

from multiprocessing.dummy import Pool
from joblib import Parallel, delayed
from sklearn.base import clone

import kenlm

from ..utils.progress import ProgressBar

import logging
logger = logging.getLogger("baseline")


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


def _getNGrams(sentence, nList):
    tokens = sentence.split()
    count = len(tokens)

    ngramsList = []

    for n in nList:
        ngramsList.append(zip(*[tokens[i:count - n + i] for i in xrange(n)]))

    return ngramsList


def _fitNGramCounts(sentences, ngramPath):
    sentences = _normalizeSentences(sentences)
    totalCount = len(sentences)

    n1s = []
    n2s = []
    n3s = []

    progress = ProgressBar(totalCount)
    for i, sentence in enumerate(sentences):
        progress.done(i)

        n1, n2, n3 = _getNGrams(sentence, [1, 2, 3])
        n1s.extend(n1)
        n2s.extend(n2)
        n2s.extend(n3)
    progress.complete()

    n1counter = collections.Counter(n1s)
    n2counter = collections.Counter(n2s)
    n3counter = collections.Counter(n3s)

    ngramCounts = (n1counter, n2counter, n3counter)

    with open(ngramPath, "wb") as ngramsFile:
        cPickle.dump(ngramCounts, ngramsFile, cPickle.HIGHEST_PROTOCOL)


def _loadNGramCounts(ngramPath):
    with open(ngramPath) as ngramsFile:
        ngramCounts = cPickle.load(ngramsFile)

    return ngramCounts


def _getHighLowFreqNGrams(counter):
    highFreqNGrams = set()
    lowFreqNGrams = set()

    totalCount = sum(counter.values())

    countTillNow = 0
    for ngram, count in counter.most_common():
        if countTillNow < totalCount / 4:
            highFreqNGrams.add(ngram)
        elif countTillNow >= 3 * totalCount / 4:
            lowFreqNGrams.add(ngram)

        countTillNow += count

    return set(highFreqNGrams), set(lowFreqNGrams)


def _getOverlapCount(sentence, ngrams, n):
    sentenceNGrams = _getNGrams(sentence, [n])[0]

    count = 0
    for ngram in sentenceNGrams:
        if ngram in ngrams:
            count += 1

    return count


def _parseSentences(sentences, parsedFilePath):
    p = Pool(10)

    parser = CoreNLPParser(
                url=os.getenv("CORENLP_HOST", "http://localhost:9000"))

    parseIterator = p.imap(lambda s: parser.parse_one(s.split()), sentences)
    parses = []

    progress = ProgressBar(len(sentences))
    for i, parse in enumerate(parseIterator):
        progress.done(i)
        parses.append(parse)
    progress.complete()

    with open(parsedFilePath, "wb") as ngramsFile:
        cPickle.dump(parses, ngramsFile, cPickle.HIGHEST_PROTOCOL)


def _loadParsedSentences(parsedFilePath):
    with open(parsedFilePath) as ngramsFile:
        parses = cPickle.load(ngramsFile)

    return parses


def _computeFeatures(srcSentences, mtSentences,
                     srcLModel, refLModel, highLowNGrams, parsePath):
    high1grams, low1grams, \
        high2grams, low2grams, \
        high3grams, low3grams = highLowNGrams

    logger.info("Loading parse trees")
    srcParses = _loadParsedSentences(parsePath)

    def _computeSentenceFeatures(srcSentence, mtSentence, srcParse):
        srcTokens = srcSentence.split()
        mtTokens = mtSentence.split()

        srcCount = float(len(srcTokens))

        features = [
            len(srcTokens),
            len(mtTokens),
            np.mean(map(len, srcTokens)),
            np.mean(map(len, mtTokens)),
            srcLModel.score(srcSentence),
            refLModel.score(mtSentence),
            float(len(mtTokens)) / float(len(set(mtTokens))),
            float(len(srcTokens)) / float(len(mtTokens)),
            float(len(mtTokens)) / float(len(srcTokens)),
            _getOverlapCount(srcSentence, low1grams, 1) / srcCount,
            _getOverlapCount(srcSentence, high1grams, 1) / srcCount,
            _getOverlapCount(srcSentence, low2grams, 2) / srcCount,
            _getOverlapCount(srcSentence, high2grams, 2) / srcCount,
            _getOverlapCount(srcSentence, low3grams, 3) / srcCount,
            _getOverlapCount(srcSentence, high3grams, 3) / srcCount,
            len(filter(lambda x: not x.isalnum(), srcTokens)),
            len(filter(lambda x: not x.isalnum(), mtTokens)),
            srcParse.height(),
        ]

        # features.extend(posCounts.transform([srcParse]).todense().tolist()[0])

        return features

    logger.info("Computing features")
    return np.array(
                map(_computeSentenceFeatures,
                    srcSentences,
                    mtSentences,
                    srcParses)
                )


def _prepareFeatures(fileBasename, devFileSuffix=None,
                     trainLM=True, trainNGrams=True, parseSentences=True):
    logger.info("Loading data for computing features")
    targetPath = fileBasename + ".hter"
    srcSentencesPath = fileBasename + ".src"
    mtSentencesPath = fileBasename + ".mt"
    refSentencesPath = fileBasename + ".ref"

    srcLMPath = fileBasename + ".src.lm.2.arpa"
    refLMPath = fileBasename + ".ref.lm.2.arpa"
    ngramPath = fileBasename + ".src.ngrams.pickle"
    srcParsePath = fileBasename + ".src.parse"
    devParsePath = fileBasename + ".src.dev.parse"

    srcSentences = _loadSentences(srcSentencesPath)
    mtSentences = _loadSentences(mtSentencesPath)
    refSentences = _loadSentences(refSentencesPath)

    y = np.clip(np.loadtxt(targetPath), 0, 1)

    if devFileSuffix:
        splitter = ShuffleSplit(n_splits=1, test_size=0, random_state=42)
        train_index, _ = splitter.split(srcSentences).next()

        srcSentencesDev = _loadSentences(srcSentencesPath + devFileSuffix)
        mtSentencesDev = _loadSentences(mtSentencesPath + devFileSuffix)

        y_dev = np.clip(np.loadtxt(targetPath + devFileSuffix), 0, 1)
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=.1, random_state=42)
        train_index, dev_index = splitter.split(srcSentences).next()

        srcSentencesDev = srcSentences[dev_index]
        mtSentencesDev = mtSentences[dev_index]

        y_dev = y[dev_index]

    srcSentencesTrain = srcSentences[train_index]
    mtSentencesTrain = mtSentences[train_index]
    refSentencesTrain = refSentences[train_index]

    y_train = y[train_index]

    if trainLM:
        logger.info("Training language models")
        _trainLM(srcSentencesTrain, srcLMPath, 2)
        _trainLM(refSentencesTrain, refLMPath, 2)

    if trainNGrams:
        logger.info("Computing ngram frequencies")
        _fitNGramCounts(srcSentencesTrain, ngramPath)

    if parseSentences:
        logger.info("Parsing sentences")
        _parseSentences(srcSentencesTrain, srcParsePath)
        _parseSentences(srcSentencesDev, devParsePath)

    # posCounts = CountVectorizer(
    #     lowercase=False,
    #     tokenizer=lambda t: map(lambda p: p[1], t.pos()),
    #     ngram_range=(1, 2)
    # )
    # posCounts.fit(srcParses)

    logger.info("Loading language models")
    srcLModel = kenlm.Model(srcLMPath)
    refLModel = kenlm.Model(refLMPath)

    logger.info("Loading ngram frequencies")
    ngramCounts = _loadNGramCounts(ngramPath)
    high1grams, low1grams = _getHighLowFreqNGrams(ngramCounts[0])
    high2grams, low2grams = _getHighLowFreqNGrams(ngramCounts[1])
    high3grams, low3grams = _getHighLowFreqNGrams(ngramCounts[2])
    highLowNGrams = (high1grams, low1grams,
                     high2grams, low2grams,
                     high3grams, low3grams)

    X_train = _computeFeatures(srcSentencesTrain, mtSentencesTrain,
                               srcLModel, refLModel, highLowNGrams,
                               srcParsePath)

    X_dev = _computeFeatures(srcSentencesDev, mtSentencesDev,
                             srcLModel, refLModel, highLowNGrams, devParsePath)

    return X_train, y_train, X_dev, y_dev


def _getFeaturesFromFile(fileBasename, devFileSuffix=None,
                         featureFileSuffix=None):
    logger.info("Loading features from file")
    targetPath = fileBasename + ".hter"

    y = np.clip(np.loadtxt(targetPath), 0, 1)
    X = np.loadtxt(fileBasename + featureFileSuffix)

    if devFileSuffix:
        splitter = ShuffleSplit(n_splits=1, test_size=0, random_state=42)
        train_index, _ = splitter.split(y).next()

        X_dev = np.loadtxt(fileBasename + featureFileSuffix + devFileSuffix)
        y_dev = np.clip(np.loadtxt(targetPath + devFileSuffix), 0, 1)
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=.1, random_state=42)
        train_index, dev_index = splitter.split(y).next()

        X_dev = X[dev_index]
        y_dev = y[dev_index]

    X_train = X[train_index]
    y_train = y[train_index]

    return X_train, y_train, X_dev, y_dev


# def plotData(X, y, svr):
#     pca = PCA(n_components=2)
#     pcaX = pca.fit_transform(X)
#
#     # X_plot = np.linspace(0, 5, 100000)[:, None]
#     # y_svr = svr.predict(X_plot)
#
#     # sv_ind = svr.support_
#     # plt.scatter(pcaX[sv_ind], y[sv_ind], c='r',
#     #             label='SVR support vectors',
#     #             zorder=2)
#     plt.scatter(pcaX[:, 0], pcaX[:, 1], c=y, cmap=cm.Oranges, label='data')
#     # plt.plot(X_plot, y_svr, c='r',  label='SVR')
#
#     # plt.xlabel('data')
#     # plt.ylabel('target')
#     # plt.title('')
#     plt.legend()
#
#     plt.show()


def _fitAndEval(svr, params, X_train, y_train, X_dev, y_dev, verbose=False):
    svr.set_params(**params)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_dev)

    result = (params, utils.evaluate(y_pred, y_dev, False))
    if verbose:
        _printResult([result])

    return result


def _printResult(results, printHeader=False):
    if printHeader:
        print "\t".join(["kernel", "C", "gamma", "MSE", "MAE",
                         "PCC", "p-value  ", "SCC", "p-value  "])

    formatString = "\t".join(["%s"] * 9)
    for row in results:
        print formatString % (
            row[0]["kernel"] if "kernel" in row[0] else "-",
            row[0]["C"] if "C" in row[0] else "-",
            row[0]["gamma"] if "gamma" in row[0] else "-",
            ("%1.5f" % row[1]["MSE"]),
            ("%1.5f" % row[1]["MAE"]),
            ("%1.5f" % row[1]["pearsonR"][0]),
            ("%.3e" % row[1]["pearsonR"][1]),
            ("%1.5f" % row[1]["spearmanR"][0]),
            ("%.3e" % row[1]["spearmanR"][1]),
        )


def train_model(workspaceDir, modelName, devFileSuffix=None,
                featureFileSuffix=None, normalize=False, tune=False,
                trainLM=True, trainNGrams=True, parseSentences=True,
                maxJobs=-1):
    logger.info("initializing TQE training")
    fileBasename = os.path.join(workspaceDir, "tqe." + modelName)

    if featureFileSuffix:
        X_train, y_train, X_dev, y_dev = _getFeaturesFromFile(
                                            fileBasename,
                                            devFileSuffix=devFileSuffix,
                                            featureFileSuffix=featureFileSuffix
                                            )
    else:
        X_train, y_train, X_dev, y_dev = _prepareFeatures(
                                            fileBasename,
                                            devFileSuffix=devFileSuffix,
                                            trainLM=trainLM,
                                            trainNGrams=trainNGrams,
                                            parseSentences=parseSentences
                                            )

    if normalize:
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_dev = scaler.transform(X_dev)

    # pca = PCA(n_components=15)
    # X = pca.fit_transform(X)

    if tune:
        parameters = [{
                'kernel': ['rbf'],
                'gamma': [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4],
                'C': [1e-2, 1e-1, 1e0, 1e1, 1e2]
            }]
    else:
        parameters = [{
            'kernel': ['rbf'],
            'gamma': [1e-2],
            'C': [1e0]
        }]

    logger.info("Training SVR")
    svr = svm.SVR(max_iter=1e7)

    results = Parallel(n_jobs=maxJobs, verbose=10)(
        delayed(_fitAndEval)(
            clone(svr), params, X_train, y_train, X_dev, y_dev
        ) for params in ParameterGrid(parameters)
    )

    logger.info("Printnig results")
    _printResult(results, True)

    # plotData(X_train, y_train, svr)


def setupArgparse(parser):
    def run(args):
        train_model(args.workspace_dir,
                    args.model_name,
                    devFileSuffix=args.dev_file_suffix,
                    featureFileSuffix=args.feature_file_suffix,
                    normalize=args.normalize,
                    tune=args.tune,
                    trainLM=args.train_lm,
                    trainNGrams=args.train_ngrams,
                    parseSentences=args.parse,
                    maxJobs=args.max_jobs)

    parser.add_argument('workspace_dir',
                        help='Directory containing prepared files')
    parser.add_argument('model_name',
                        help='Identifier for prepared files used with ' +
                        'preparation')
    parser.add_argument('--dev-file-suffix', type=str, default=None,
                        help='Suffix for test files')
    parser.add_argument('--feature-file-suffix', type=str, default=None,
                        help='Suffix for feature files')
    parser.add_argument('--train-lm', action='store_true',
                        help='Train language model.')
    parser.add_argument('--train-ngrams', action='store_true',
                        help='Compute ngram freqs.')
    parser.add_argument('--parse', action='store_true',
                        help='Parse sentences.')
    parser.add_argument('--normalize', action='store_true',
                        help='Weather to normalize features or not.')
    parser.add_argument('--tune', action='store_true',
                        help='Weather to tune parameters or not.')
    parser.add_argument('--max-jobs', type=int, default=-1,
                        help='Maximum number of jobs to run parallelly')
    parser.set_defaults(func=run)
