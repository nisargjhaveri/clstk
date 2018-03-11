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


def _preprocessSentences(sentences, lower=True, tokenize=True):
    def _processSentence(sentece):
        sentence = sentece.strip()

        if lower:
            sentence = sentence.lower()

        if tokenize:
            sentence = np.array(sentence.split(), dtype=object)

        return sentence

    return np.array(map(_processSentence, sentences), dtype=object)


def _loadSentences(filePath, lower=True, tokenize=True):
    with open(filePath) as lines:
        lines = map(lambda l: l.decode('utf-8'), list(lines))
        sentences = _preprocessSentences(lines,
                                         lower=lower, tokenize=tokenize)

    return sentences


def _loadData(fileBasename, devFileSuffix=None, testFileSuffix=None,
              lower=True, tokenize=True):
    targetPath = fileBasename + ".hter"
    srcSentencesPath = fileBasename + ".src"
    mtSentencesPath = fileBasename + ".mt"
    refSentencesPath = fileBasename + ".ref"

    srcSentences = _loadSentences(srcSentencesPath, lower, tokenize)
    mtSentences = _loadSentences(mtSentencesPath, lower, tokenize)
    refSentences = _loadSentences(refSentencesPath, lower, tokenize)

    y = np.clip(np.loadtxt(targetPath), 0, 1)

    if (testFileSuffix or devFileSuffix) and \
            not (testFileSuffix and devFileSuffix):
        raise ValueError("You have to specify both dev and test file suffix")

    if devFileSuffix and testFileSuffix:
        splitter = ShuffleSplit(n_splits=1, test_size=0, random_state=42)
        train_index, _ = splitter.split(srcSentences).next()

        srcSentencesDev = _loadSentences(srcSentencesPath + devFileSuffix,
                                         lower, tokenize)
        mtSentencesDev = _loadSentences(mtSentencesPath + devFileSuffix,
                                        lower, tokenize)
        refSentencesDev = _loadSentences(refSentencesPath + devFileSuffix,
                                         lower, tokenize)

        srcSentencesTest = _loadSentences(srcSentencesPath + testFileSuffix,
                                          lower, tokenize)
        mtSentencesTest = _loadSentences(mtSentencesPath + testFileSuffix,
                                         lower, tokenize)
        refSentencesTest = _loadSentences(refSentencesPath + testFileSuffix,
                                          lower, tokenize)

        y_dev = np.clip(np.loadtxt(targetPath + devFileSuffix), 0, 1)
        y_test = np.clip(np.loadtxt(targetPath + testFileSuffix), 0, 1)
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=.2, random_state=42)
        train_index, dev_index = splitter.split(srcSentences).next()

        dev_len = len(dev_index) / 2

        srcSentencesDev = srcSentences[dev_index[:dev_len]]
        mtSentencesDev = mtSentences[dev_index[:dev_len]]
        refSentencesDev = refSentences[dev_index[:dev_len]]

        srcSentencesTest = srcSentences[dev_index[dev_len:]]
        mtSentencesTest = mtSentences[dev_index[dev_len:]]
        refSentencesTest = refSentences[dev_index[dev_len:]]

        y_dev = y[dev_index[:dev_len]]
        y_test = y[dev_index[dev_len:]]

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
    X_test = {
        "src": srcSentencesTest,
        "mt": mtSentencesTest,
        "ref": refSentencesTest
    }

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def pad_sequences(sequences, maxlen=None, **kwargs):
    if maxlen <= 0:
        return sequences
    else:
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(sequences, maxlen, **kwargs)


def getBatchGenerator(*args, **kwargs):
    """
    X is assumed to be list of inputs
    y is assumed to be list of outputs
    """
    from keras.utils import Sequence

    class BatchGeneratorSequence(Sequence):
        def __init__(self, X, y=None, key=lambda x: x, batch_size=None):
            self.batch_size = batch_size
            self.emit_y = bool(y)

            groupingKyes = map(key, zip(*X))

            self.alignment = []
            groups = {}
            for i, key in enumerate(groupingKyes):
                groups.setdefault(key, []).append(i)
                self.alignment.append(i)

            self.batches = []
            for group in groups.values():
                if not batch_size:
                    X_batches = [[np.array(x_i[group].tolist())
                                  for x_i in X]]
                    y_batches = [[np.array(y_i[group].tolist())
                                  for y_i in y] if y else None]
                else:
                    num_samples = len(group)
                    X_batches = [[np.array(
                                        x_i[group[i:i + batch_size]].tolist())
                                  for x_i in X]
                                 for i in xrange(0, num_samples, batch_size)]
                    y_batches = [[np.array(
                                        y_i[group[i:i + batch_size]].tolist())
                                  for y_i in y] if y else None
                                 for i in xrange(0, num_samples, batch_size)]
                self.batches.extend(zip(X_batches, y_batches))

        def __len__(self):
            # print len(self.batches)
            return len(self.batches)

        def __getitem__(self, idx):
            return self.batches[idx] if self.emit_y else self.batches[idx][0]

        def align(self, y):
            return y[self.alignment]

    return BatchGeneratorSequence(*args, **kwargs)


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


def getStatefulPearsonr():
    from keras.layers import Layer
    import keras.backend as K

    class StatefulPearsonr(Layer):
        def __init__(self, **kwargs):
            super(StatefulPearsonr, self).__init__(**kwargs)

            self.n = K.variable(value=0, dtype='int')
            self.sum_xy = K.variable(value=0, dtype='float')
            self.sum_x = K.variable(value=0, dtype='float')
            self.sum_y = K.variable(value=0, dtype='float')
            self.sum_x_2 = K.variable(value=0, dtype='float')
            self.sum_y_2 = K.variable(value=0, dtype='float')

        def reset_states(self):
            K.set_value(self.n, 0)
            K.set_value(self.sum_xy, 0)
            K.set_value(self.sum_x, 0)
            K.set_value(self.sum_y, 0)
            K.set_value(self.sum_x_2, 0)
            K.set_value(self.sum_y_2, 0)

        def __call__(self, y_true, y_pred):
            x = y_true
            y = y_pred
            # n = x.shape[0]
            mx = K.mean(x)
            my = K.mean(y)
            xm, ym = x - mx, y - my
            r_num = K.sum(xm * ym)
            r_den = K.sqrt(K.sum(xm * xm) * K.sum(ym * ym))
            r = r_num / r_den

            n = self.n + K.shape(x)[0]
            sum_xy = self.sum_xy + K.sum(x * y)
            sum_x = self.sum_x + K.sum(x)
            sum_y = self.sum_y + K.sum(y)
            sum_x_2 = self.sum_x_2 + K.sum(x * x)
            sum_y_2 = self.sum_y_2 + K.sum(y * y)

            self.add_update(K.update_add(self.n, K.shape(x)[0]),
                            inputs=[y_true, y_pred])
            self.add_update(K.update_add(self.sum_xy, K.sum(x * y)),
                            inputs=[y_true, y_pred])
            self.add_update(K.update_add(self.sum_x, K.sum(x)),
                            inputs=[y_true, y_pred])
            self.add_update(K.update_add(self.sum_y, K.sum(y)),
                            inputs=[y_true, y_pred])
            self.add_update(K.update_add(self.sum_x_2, K.sum(x * x)),
                            inputs=[y_true, y_pred])
            self.add_update(K.update_add(self.sum_y_2, K.sum(y * y)),
                            inputs=[y_true, y_pred])

            r_num = (n * sum_xy) - (sum_x * sum_y)
            r_den = (K.sqrt((n * sum_x_2) - (sum_x * sum_x))
                     * K.sqrt((n * sum_y_2) - (sum_y * sum_y)))
            r = r_num / r_den

            # Presumably, if abs(r) > 1, then it is only some small artifact of
            # floating point arithmetic.
            r = K.clip(r, -1.0, 1.0)
            return r

    return StatefulPearsonr()
