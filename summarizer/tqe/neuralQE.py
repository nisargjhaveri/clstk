import os

import numpy as np
from sklearn.model_selection import ShuffleSplit

from keras.layers import Input, Embedding, Dense
from keras.layers import RNN, GRU, GRUCell, TimeDistributed, Bidirectional
from keras.models import Model

from keras.preprocessing.sequence import pad_sequences
from keras.utils.generic_utils import CustomObjectScope


import logging
logger = logging.getLogger("neuralQE")


class WordIndexTransformer(object):
    def __init__(self):
        self.nextIndex = 1
        self.vocabMap = {}

    def _getIndex(self, token):
        return self.vocabMap.get(token, 0)

    def fit(self, sentences):
        for sentence in sentences:
            for token in sentence:
                if token not in self.vocabMap:
                    self.vocabMap[token] = self.nextIndex
                    self.nextIndex += 1

    def transform(self, sentences):
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


def _prepareInput(fileBasename, srcVocabTransformer, refVocabTransformer,
                  devFileSuffix=None):
    logger.info("Loading data")
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

    logger.info("Transforming sentences to onehot")

    srcSentencesTrain = srcVocabTransformer.fit_transform(srcSentencesTrain)
    srcSentencesDev = srcVocabTransformer.fit_transform(srcSentencesDev)

    mtSentencesTrain = refVocabTransformer.fit_transform(mtSentencesTrain)
    mtSentencesDev = refVocabTransformer.fit_transform(mtSentencesDev)
    refSentencesTrain = refVocabTransformer.fit_transform(refSentencesTrain)
    refSentencesDev = refVocabTransformer.fit_transform(refSentencesDev)

    X_train = {
        "src": pad_sequences(srcSentencesTrain),
        "mt": pad_sequences(mtSentencesTrain),
        "ref": pad_sequences(refSentencesTrain)
    }

    X_dev = {
        "src": pad_sequences(srcSentencesDev),
        "mt": pad_sequences(mtSentencesDev),
        "ref": pad_sequences(refSentencesDev)
    }

    return X_train, y_train, X_dev, y_dev


class AttentionGRUCell(GRUCell):
    def __init__(self, *args, **kwargs):
        kwargs.get('attention')
        super(AttentionGRUCell, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        # constants_shape = None
        if isinstance(input_shape, list):
            # if len(input_shape) > 1:
            #     constants_shape = input_shape[1:]
            input_shape = input_shape[0]

        super(AttentionGRUCell, self).build(input_shape)

    def call(self, inputs, states, training=None, constants=None):
        return super(AttentionGRUCell, self).call(inputs, states,
                                                  training=training)


def _printModelSummary(model):
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')

    model_summary = ["Printing model summary"]

    def summary_capture(line):
        model_summary.append(line)

    model.summary(print_fn=summary_capture)
    logger.info("\n".join(model_summary))


def getModel(srcVocabTransformer, refVocabTransformer):
    embedding_size = 300
    hidden_size = 2
    # num_layers = 3
    # time_stamps = 5

    src_vocab_size = srcVocabTransformer.vocab_size()
    ref_vocab_size = refVocabTransformer.vocab_size()

    logger.info("Creating model")

    src_input = Input(shape=(None, ))
    ref_input = Input(shape=(None, ))

    src_embedding = Embedding(
                        output_dim=embedding_size,
                        input_dim=src_vocab_size,
                        mask_zero=True)(src_input)

    ref_embedding = Embedding(
                        output_dim=embedding_size,
                        input_dim=ref_vocab_size,
                        mask_zero=True)(ref_input)

    encoder = Bidirectional(
                    GRU(hidden_size, return_sequences=True, return_state=True)
            )(src_embedding)

    with CustomObjectScope({'AttentionGRUCell': AttentionGRUCell}):
        decoder = Bidirectional(
                    RNN(AttentionGRUCell(hidden_size), return_sequences=True)
                )(
                    ref_embedding,
                    initial_state=encoder[1:],
                    constants=encoder[0]
                )

    out = TimeDistributed(
        Dense(ref_vocab_size, activation='softmax')
    )(decoder)

    model = Model(inputs=[src_input, ref_input], outputs=out)

    logger.info("Compiling model")
    model.compile("adagrad", "sparse_categorical_crossentropy")

    _printModelSummary(model)

    return model


def train_model(workspaceDir, modelName, devFileSuffix=None):
    logger.info("initializing TQE training")
    fileBasename = os.path.join(workspaceDir, "tqe." + modelName)

    srcVocabTransformer = WordIndexTransformer()
    refVocabTransformer = WordIndexTransformer()

    X_train, y_train, X_dev, y_dev = _prepareInput(
                                        fileBasename,
                                        srcVocabTransformer,
                                        refVocabTransformer,
                                        devFileSuffix=devFileSuffix,
                                        )

    model = getModel(srcVocabTransformer, refVocabTransformer)

    logger.info("Training")
    model.fit([
        X_train['src'],
        X_train['ref']
    ], [
        X_train['ref'].reshape((len(X_train['ref']), -1, 1)),
    ], batch_size=200, epochs=5)

    logger.info("Predicting")
    # print model.predict([
    #     np.array([[1, 2, 3, 1, 1], [1, 2, 1, 3, 1]]),
    #     np.array([[1, 1, 1, 1, 1]] * 2)
    # ])


def setupArgparse(parser):
    def run(args):
        train_model(args.workspace_dir,
                    args.model_name,
                    devFileSuffix=args.dev_file_suffix)

    parser.add_argument('workspace_dir',
                        help='Directory containing prepared files')
    parser.add_argument('model_name',
                        help='Identifier for prepared files used with ' +
                        'preparation')
    parser.add_argument('--dev-file-suffix', type=str, default=None,
                        help='Suffix for test files')
    parser.set_defaults(func=run)
