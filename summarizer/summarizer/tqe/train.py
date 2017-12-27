import os
import subprocess

import regex

import numpy as np
from sklearn import svm
from sklearn.model_selection import ShuffleSplit

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


def _train_lm(sentences, lmFilePath, order):
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
        lmplz.communicate("\n".join(sentences).encode("utf-8"))
        devNull is None or devNull.close()


def _getFeatures(sourceSentences, targetSentences, sourceLMPath, targetLMPath):
    # logger.info("Loading language models")
    # sourceModel = kenlm.Model(sourceLMPath)
    # targetModel = kenlm.Model(targetLMPath)

    def _computeSentenceFeatures(sourceSentence, targetSentence):
        sourceTokens = sourceSentence.split()
        targetTokens = targetSentence.split()

        punc = regex.compile(r'[^\w]', regex.UNICODE)

        return [
            len(sourceTokens),
            # len(targetTokens),
            np.mean(map(len, sourceTokens)),
            np.mean(map(len, targetTokens)),
            # float(len(sourceTokens)) / float(len(targetTokens)),
            # float(len(targetTokens)) / float(len(sourceTokens)),
            len(filter(punc.search, sourceTokens)),
            # len(filter(punc.search, targetTokens)),
            float(len(set(targetTokens))) / float(len(targetTokens)),
            # sourceModel.score(sourceSentence),
            # targetModel.score(targetSentence),
        ]

    logger.info("Computing features")
    return np.array(
                map(_computeSentenceFeatures,
                    sourceSentences,
                    targetSentences)
                )


def train_model(workspaceDir, modelName, evaluate=False, trainLM=True):
    logger.info("initializing TQE training")
    fileBasename = os.path.join(workspaceDir, "tqe." + modelName)

    targetPath = fileBasename + ".hter"
    srcSentencesPath = fileBasename + ".src"
    mtSentencesPath = fileBasename + ".mt"
    refSentencesPath = fileBasename + ".ref"

    srcLMPath = fileBasename + ".src.lm.2.arpa"
    refLMPath = fileBasename + ".ref.lm.2.arpa"

    srcSentences = _loadSentences(srcSentencesPath)
    mtSentences = _loadSentences(mtSentencesPath)
    refSentences = _loadSentences(refSentencesPath)

    if evaluate:
        splitter = ShuffleSplit(n_splits=1, test_size=.1, random_state=42)
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=0, random_state=42)

    for train_index, test_index in splitter.split(srcSentences):
        if trainLM:
            logger.info("Training language models")
            _train_lm(srcSentences[train_index], srcLMPath, 2)
            _train_lm(refSentences[train_index], refLMPath, 2)

        y = np.loadtxt(targetPath)
        X = _getFeatures(srcSentences, mtSentences, srcLMPath, refLMPath)

        X_train = X[train_index]
        y_train = y[train_index]

        logger.info("Training SVR")
        svr = svm.SVR(verbose=True)
        svr.fit(X_train, y_train)

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
