import os

import numpy as np
from sklearn.model_selection import ShuffleSplit

from keras.layers import Input, Embedding, Dense, Layer, Reshape
from keras.layers import RNN, GRU, GRUCell, TimeDistributed, Bidirectional
from keras.layers import MaxPooling1D
from keras.models import Model

import keras.backend as K

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
    def __init__(self, units, *args, **kwargs):
        super(AttentionGRUCell, self).__init__(units, *args, **kwargs)

    def build(self, input_shape):
        self.constants_shape = None
        if isinstance(input_shape, list):
            if len(input_shape) > 1:
                self.constants_shape = input_shape[1:]
            input_shape = input_shape[0]

        cell_input_shape = list(input_shape)
        cell_input_shape[-1] += self.constants_shape[0][-1]
        cell_input_shape = tuple(cell_input_shape)

        super(AttentionGRUCell, self).build(cell_input_shape)

    def attend(self, query, attention_states):
        # Multiply query with each state per batch
        attention = K.batch_dot(
                        attention_states, query,
                        axes=(attention_states.ndim - 1, query.ndim - 1)
                    )

        # Take softmax to get weight per timestamp
        attention = K.softmax(attention)

        # Take weigthed average of attention_states
        context = K.batch_dot(attention, attention_states)

        return context

    def call(self, inputs, states, training=None, constants=None):
        context = self.attend(states[0], constants[0])

        inputs = K.concatenate([context, inputs])

        cell_out, cell_state = super(AttentionGRUCell, self).call(
                                            inputs, states, training=training)

        return cell_out, cell_state


class AlignStates(Layer):
    def __init__(self, **kwargs):
        super(AlignStates, self).__init__(**kwargs)

        self.supports_masking = True  # TODO FIXME

    def rightShift(self, x):
        return K.concatenate(
            [
                K.zeros_like(x[:, -1:]),
                x[:, :-1]
            ],
            axis=1
        )

    def leftShift(self, x):
        return K.concatenate(
            [
                x[:, 1:],
                K.zeros_like(x[:, :1])
            ],
            axis=1
        )

    def call(self, x, mask=None):
        decoder_for, decoder_back, ref_embedding = x
        s = K.concatenate([
            self.rightShift(decoder_for),
            self.leftShift(decoder_back),
        ])
        e = K.concatenate([
            self.rightShift(ref_embedding),
            self.leftShift(ref_embedding),
        ])
        return K.concatenate([s, e])

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1],
                (input_shape[0][2] + input_shape[2][2]) * 2)


def TimeDistributedSequential(layers, inputs):
    input = inputs
    for layer in layers:
        input = TimeDistributed(
                    layer, name="_".join(["td", layer.name])
                )(input)
    return input


def _printModelSummary(model):
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')

    model_summary = ["Printing model summary"]

    def summary_capture(line):
        model_summary.append(line)

    model.summary(print_fn=summary_capture)
    logger.info("\n".join(model_summary))


def getModel(srcVocabTransformer, refVocabTransformer,
             embedding_size,
             gru_size,
             qualvec_size,
             maxout_size,
             maxout_units):
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
                    GRU(gru_size, return_sequences=True, return_state=True)
            )(src_embedding)

    attention_states = TimeDistributedSequential(
                            [Dense(gru_size)],
                            encoder[0]
                        )

    with CustomObjectScope({'AttentionGRUCell': AttentionGRUCell}):
        decoder = Bidirectional(
                    RNN(AttentionGRUCell(gru_size), return_sequences=True),
                    merge_mode=None
                )(
                    ref_embedding,
                    initial_state=encoder[1:],
                    constants=attention_states
                )

    alignedStates = AlignStates()([decoder[0], decoder[1], ref_embedding])

    out = TimeDistributedSequential([
        Dense(maxout_size * maxout_units),  # t_tilda
        Reshape((-1, 1)),  # Reshaping for maxout to work
        MaxPooling1D(maxout_units),  # Maxout
        Reshape((-1,)),  # t
        Dense(qualvec_size),  # t * W_o2
        Dense(ref_vocab_size, activation="softmax"),
    ], alignedStates)

    model = Model(inputs=[src_input, ref_input], outputs=[out])

    logger.info("Compiling model")
    model.compile(
            optimizer="adagrad",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    _printModelSummary(model)

    return model


def train_model(workspaceDir, modelName, devFileSuffix=None,
                batchSize=50, epochs=15, **kwargs):
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

    model = getModel(srcVocabTransformer, refVocabTransformer, **kwargs)

    logger.info("Training")
    model.fit([
            X_train['src'],
            X_train['ref']
        ], [
            X_train['ref'].reshape((len(X_train['ref']), -1, 1)),
        ],
        batch_size=batchSize,
        epochs=epochs,
        validation_data=([
                X_dev['src'],
                X_dev['ref']
            ], [
                X_dev['ref'].reshape((len(X_dev['ref']), -1, 1)),
            ]
        ))

    logger.info("Saving model")
    model.save(fileBasename + "neural.model.h5")

    logger.info("Predicting")
    print X_dev['ref']
    print model.predict([
        X_dev['src'],
        X_dev['ref']
    ])


def setupArgparse(parser):
    def run(args):
        train_model(args.workspace_dir,
                    args.model_name,
                    devFileSuffix=args.dev_file_suffix,
                    batchSize=args.batch_size,
                    epochs=args.epochs,
                    embedding_size=args.embedding_size,
                    gru_size=args.gru_size,
                    qualvec_size=args.qualvec_size,
                    maxout_size=args.maxout_size,
                    maxout_units=args.maxout_units
                    )

    parser.add_argument('workspace_dir',
                        help='Directory containing prepared files')
    parser.add_argument('model_name',
                        help='Identifier for prepared files')
    parser.add_argument('--dev-file-suffix', type=str, default=None,
                        help='Suffix for test files')
    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=15,
                        help='Number of epochs to run')
    parser.add_argument('-m', '--embedding-size', type=int, default=300,
                        help='Size of word embeddings')
    parser.add_argument('-n', '--gru-size', type=int, default=500,
                        help='Size of GRU')
    parser.add_argument('-q', '--qualvec-size', type=int, default=500,
                        help='Size of last layer connected before softmax')
    parser.add_argument('-l', '--maxout-size', type=int, default=500,
                        help='Size of maxout layer output')
    parser.add_argument('--maxout-units', type=int, default=2,
                        help='Number of maxout units')
    parser.set_defaults(func=run)
