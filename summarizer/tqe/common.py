from collections import Counter

import numpy as np
from sklearn.model_selection import ShuffleSplit


class WordIndexTransformer(object):
    def __init__(self, vocab_size=None):
        self.nextIndex = 1
        self.vocabSize = vocab_size
        self.vocabMap = {}
        self.wordCounts = Counter()

        self.finalized = False

    def _getIndex(self, token):
        return self.vocabMap.get(token, 0)

    def fit(self, sentences):
        if self.finalized:
            raise ValueError("Cannot fit after the transformer is finalized")

        for sentence in sentences:
            self.wordCounts.update(sentence)

        return self

    def finalize(self):
        if self.finalized:
            return

        for token, count in self.wordCounts.most_common():
            if self.vocabSize is None or self.nextIndex <= self.vocabSize:
                self.vocabMap[token] = self.nextIndex
                self.nextIndex += 1

        self.finalized = True

        return self

    def transform(self, sentences):
        self.finalize()

        transformedSentences = []
        for sentence in sentences:
            transformedSentences.append(
                np.array([self._getIndex(token) for token in sentence]))

        return np.array(transformedSentences)

    def fit_transform(self, sentences):
        self.fit(sentences)
        return self.transform(sentences)

    def vocab_size(self):
        return self.nextIndex

    def vocab_map(self):
        return self.vocabMap


def _loadSentences(filePath, lower=True, tokenize=True):
    def _processLine(line):
        sentence = line.decode('utf-8').strip()

        if lower:
            sentence = sentence.lower()

        if tokenize:
            sentence = np.array(sentence.split(), dtype=object)

        return sentence

    with open(filePath) as lines:
        sentences = map(_processLine, list(lines))

    return np.array(sentences, dtype=object)


def _loadData(fileBasename, devFileSuffix=None):
    targetPath = fileBasename + ".hter"
    srcSentencesPath = fileBasename + ".src"
    mtSentencesPath = fileBasename + ".mt"
    refSentencesPath = fileBasename + ".ref"

    srcSentences = _loadSentences(srcSentencesPath)
    mtSentences = _loadSentences(mtSentencesPath)
    refSentences = _loadSentences(refSentencesPath)

    y = np.clip(np.loadtxt(targetPath), 0, 1)

    if devFileSuffix:
        splitter = ShuffleSplit(n_splits=1, test_size=0, random_state=42)
        train_index, _ = splitter.split(srcSentences).next()

        srcSentencesDev = _loadSentences(srcSentencesPath + devFileSuffix)
        mtSentencesDev = _loadSentences(mtSentencesPath + devFileSuffix)
        refSentencesDev = _loadSentences(refSentencesPath + devFileSuffix)

        y_dev = np.clip(np.loadtxt(targetPath + devFileSuffix), 0, 1)
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=.1, random_state=42)
        train_index, dev_index = splitter.split(srcSentences).next()

        srcSentencesDev = srcSentences[dev_index]
        mtSentencesDev = mtSentences[dev_index]
        refSentencesDev = refSentences[dev_index]

        y_dev = y[dev_index]

    srcSentencesTrain = srcSentences[train_index]
    mtSentencesTrain = mtSentences[train_index]
    refSentencesTrain = refSentences[train_index]

    y_train = y[train_index]

    X_train = {
        "src": srcSentencesTrain,
        "mt": mtSentencesTrain,
        "ref": refSentencesTrain
    }
    X_dev = {
        "src": srcSentencesDev,
        "mt": mtSentencesDev,
        "ref": refSentencesDev
    }

    return X_train, y_train, X_dev, y_dev


def get_fastText_embeddings(fastText_file, vocabTransformer, embedding_size):
    import fastText
    ft_model = fastText.load_model(fastText_file)

    embedding_matrix = np.zeros(
                        shape=(vocabTransformer.vocab_size(), embedding_size)
                    )

    for token, i in vocabTransformer.vocab_map().items():
        embedding_matrix[i] = ft_model.get_word_vector(token)

    return embedding_matrix


def TimeDistributedSequential(layers, inputs, name=None):
    from keras.layers import TimeDistributed

    layer_names = ["_".join(["td", layer.name]) for layer in layers]

    if name:
        layer_names[-1] = name

    input = inputs
    for layer, layer_name in zip(layers, layer_names):
        input = TimeDistributed(
                    layer, name=layer_name
                )(input)

    return input


def _printModelSummary(logger, model, name, plot=False):
    if plot:
        from keras.utils import plot_model
        plot_model(model, to_file=(name if name else "model") + ".png")

    model_summary = ["Printing model summary"]

    if name:
        model_summary += ["Model " + name]

    def summary_capture(line):
        model_summary.append(line)

    model.summary(print_fn=summary_capture)
    logger.info("\n".join(model_summary))


def pearsonr(y_true, y_pred):
    # From https://github.com/scipy/scipy/blob/v0.14.0/scipy/stats/stats.py
    #
    # x = np.asarray(x)
    # y = np.asarray(y)
    # n = len(x)
    # mx = x.mean()
    # my = y.mean()
    # xm, ym = x-mx, y-my
    # r_num = np.add.reduce(xm * ym)
    # r_den = np.sqrt(ss(xm) * ss(ym))
    # r = r_num / r_den
    #
    # # Presumably, if abs(r) > 1, then it is only some small artifact of
    # # floating point arithmetic.
    # r = max(min(r, 1.0), -1.0)
    # df = n-2
    # if abs(r) == 1.0:
    #     prob = 0.0
    # else:
    #     t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    #     prob = betai(0.5*df, 0.5, df / (df + t_squared))
    # return r, prob

    import keras.backend as K

    x = y_true
    y = y_pred
    # n = x.shape[0]
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(xm * xm) * K.sum(ym * ym))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    r = K.clip(r, -1.0, 1.0)
    # df = n-2
    # if abs(r) == 1.0:
    #     prob = 0.0
    # else:
    #     t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    #     prob = betai(0.5*df, 0.5, df / (df + t_squared))
    return r
