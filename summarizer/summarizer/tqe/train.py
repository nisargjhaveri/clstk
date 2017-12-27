import argparse

import regex

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

import kenlm


def _loadSentences(filePath):
    with open(filePath) as lines:
        sentences = map(
            lambda s: s.decode('utf-8').strip(), list(lines))

    return sentences


def _getFeatures(sourceSentences, targetSentences, sourceLMPath, targetLMPath):
    sourceModel = kenlm.Model(sourceLMPath)
    targetModel = kenlm.Model(targetLMPath)

    def _computeSentenceFeatures(sourceSentence, targetSentence):
        sourceTokens = sourceSentence.split()
        targetTokens = targetSentence.split()

        punc = regex.compile(r'[^\w]', regex.UNICODE)

        return [
            len(sourceTokens),
            len(targetTokens),
            np.mean(map(len, sourceTokens)),
            np.mean(map(len, targetTokens)),
            float(len(sourceTokens)) / float(len(targetTokens)),
            float(len(targetTokens)) / float(len(sourceTokens)),
            len(filter(punc.search, sourceTokens)),
            len(filter(punc.search, targetTokens)),
            float(len(set(targetTokens))) / float(len(targetTokens)),
            sourceModel.score(sourceSentence),
            targetModel.score(targetSentence),
        ]

    return np.array(
                map(_computeSentenceFeatures,
                    sourceSentences,
                    targetSentences)
                )


def _train_model(targetPath, sourceSentencesPath, targetSentencesPath,
                 sourceLMPath, targetLMPath, evaluate=False):
    srcSentences = _loadSentences(sourceSentencesPath)
    mtSentences = _loadSentences(sourceSentencesPath)

    y = np.loadtxt(targetPath)
    X = _getFeatures(srcSentences, mtSentences, sourceLMPath, targetLMPath)

    if evaluate:
        X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.1, random_state=42)
    else:
        X_train, y_train = X, y

    svr = svm.SVR(verbose=True)
    svr.fit(X_train, y_train)

    if evaluate:
        y_pred = svr.predict(X_test)
        _evaluate(y_pred, y_test)


def _evaluate(y_pred, y_test):
    print "MSE:", mean_squared_error(y_test, y_pred)
    print "MAE:", mean_absolute_error(y_test, y_pred)
    print "Pearson's r:", pearsonr(y_pred, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Train Translation Quality Estimation')
    parser.add_argument('target_path',
                        help='File containing target scores.')
    parser.add_argument('source_sentences',
                        help='File containing source sentences.')
    parser.add_argument('target_sentences',
                        help='File containing MT sentences.')
    parser.add_argument('source_lm',
                        help='Langauge model path for source language')
    parser.add_argument('target_lm',
                        help='Langauge model path for target language')
    parser.add_argument('--evaluate', action='store_true',
                        help='Also evaluate the trained model.')

    args = parser.parse_args()

    _train_model(args.target_path,
                 args.source_sentences,
                 args.target_sentences,
                 args.source_lm,
                 args.target_lm,
                 args.evaluate)
