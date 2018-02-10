import os

from . import utils

import numpy as np
from sklearn.model_selection import ShuffleSplit

from keras.layers import Layer, multiply
from keras.layers import Input, Embedding, Dense, Reshape
from keras.layers import RNN, GRU, GRUCell, TimeDistributed, Bidirectional
from keras.layers import MaxPooling1D
from keras.models import Model

import keras.backend as K

from keras.preprocessing.sequence import pad_sequences
from keras.utils.generic_utils import CustomObjectScope

from collections import Counter


import logging
logger = logging.getLogger("neuralQE")


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

    srcVocabTransformer \
        .fit(srcSentencesTrain) \
        .fit(srcSentencesDev)

    srcSentencesTrain = srcVocabTransformer.transform(srcSentencesTrain)
    srcSentencesDev = srcVocabTransformer.transform(srcSentencesDev)

    refVocabTransformer.fit(mtSentencesTrain) \
                       .fit(mtSentencesDev) \
                       .fit(refSentencesTrain) \
                       .fit(refSentencesDev)

    mtSentencesTrain = refVocabTransformer.transform(mtSentencesTrain)
    mtSentencesDev = refVocabTransformer.transform(mtSentencesDev)
    refSentencesTrain = refVocabTransformer.transform(refSentencesTrain)
    refSentencesDev = refVocabTransformer.transform(refSentencesDev)

    def getMaxLen(listOfsequences):
        return max([max(map(len, sequences)) for sequences in listOfsequences])

    srcMaxLen = getMaxLen([srcSentencesTrain, srcSentencesDev])
    refMaxLen = getMaxLen([mtSentencesTrain, mtSentencesDev,
                           refSentencesTrain, refSentencesDev])

    X_train = {
        "src": pad_sequences(srcSentencesTrain, maxlen=srcMaxLen),
        "mt": pad_sequences(mtSentencesTrain, maxlen=refMaxLen),
        "ref": pad_sequences(refSentencesTrain, maxlen=refMaxLen)
    }

    X_dev = {
        "src": pad_sequences(srcSentencesDev, maxlen=srcMaxLen),
        "mt": pad_sequences(mtSentencesDev, maxlen=refMaxLen),
        "ref": pad_sequences(refSentencesDev, maxlen=refMaxLen)
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


class DenseTransposeEmbedding(Layer):
    def __init__(self, layer, units, mask_zero, **kwargs):
        self.layer = layer
        self.units = units
        self.mask_zero = mask_zero

        super(DenseTransposeEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        assert self.layer.built

        self._trainable_weights.append(self.layer.kernel)

        super(DenseTransposeEmbedding, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        out = K.gather(self.layer.kernel.transpose(), inputs)
        return out

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.units,)

    def get_config(self):
        config = {
            'layer': self.layer,
            'units': self.units,
            'mask_zero': self.mask_zero
        }
        base_config = super(DenseTransposeEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def TimeDistributedSequential(layers, inputs, name=None):
    layer_names = ["_".join(["td", layer.name]) for layer in layers]

    if name:
        layer_names[-1] = name

    input = inputs
    for layer, layer_name in zip(layers, layer_names):
        input = TimeDistributed(
                    layer, name=layer_name
                )(input)

    return input


def _printModelSummary(model, name):
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')

    model_summary = ["Printing model summary"]

    if name:
        model_summary += ["Model " + name]

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
                        mask_zero=True,
                        name="src_embedding")(src_input)

    ref_embedding = Embedding(
                        output_dim=embedding_size,
                        input_dim=ref_vocab_size,
                        mask_zero=True,
                        name="ref_embedding")(ref_input)

    encoder = Bidirectional(
                    GRU(gru_size, return_sequences=True, return_state=True),
                    name="encoder"
            )(src_embedding)

    attention_states = TimeDistributedSequential(
                            [Dense(gru_size, name="attention_state")],
                            encoder[0]
                        )

    with CustomObjectScope({'AttentionGRUCell': AttentionGRUCell}):
        decoder = Bidirectional(
                    RNN(AttentionGRUCell(gru_size), return_sequences=True),
                    merge_mode=None,
                    name="decoder"
                )(
                    ref_embedding,
                    initial_state=encoder[1:],
                    constants=attention_states
                )

    alignedStates = AlignStates()([decoder[0], decoder[1], ref_embedding])

    out_state = TimeDistributedSequential([
        Dense(maxout_size * maxout_units, name="t_tilda"),  # t_tilda
        Reshape((-1, 1)),  # Reshaping for maxout to work
        MaxPooling1D(maxout_units),  # Maxout
        Reshape((-1,)),  # t
        Dense(qualvec_size, name="t_out"),  # t * W_o2
    ], alignedStates)

    out_embeddings = Dense(ref_vocab_size, use_bias=False,
                           activation='softmax')

    predicted_word = TimeDistributedSequential([
        out_embeddings
    ], out_state, name="predicted_word")

    # Extract Quality Vectors
    W_y = DenseTransposeEmbedding(out_embeddings, qualvec_size,
                                  mask_zero=True, name="W_y")(ref_input)

    qualvec = multiply([out_state, W_y])

    quality_summary = Bidirectional(GRU(gru_size), name="estimator")(qualvec)

    quality = Dense(1, name="quality")(quality_summary)

    logger.info("Compiling model")
    model_multitask = Model(inputs=[src_input, ref_input],
                            outputs=[predicted_word, quality])
    model_multitask.compile(
            optimizer="adagrad",
            loss={
                "predicted_word": "sparse_categorical_crossentropy",
                "quality": "mse"
            },
            metrics={
                "predicted_word": ["sparse_categorical_accuracy"],
                "quality": ["mse", "mae"]
            }
        )
    _printModelSummary(model_multitask, "model_multitask")

    model_predictor = Model(inputs=[src_input, ref_input],
                            outputs=[predicted_word])
    model_predictor.compile(
            optimizer="adagrad",
            loss={
                "predicted_word": "sparse_categorical_crossentropy",
            },
            metrics={
                "predicted_word": ["sparse_categorical_accuracy"],
            }
        )
    _printModelSummary(model_predictor, "model_predictor")

    model_estimator = Model(inputs=[src_input, ref_input],
                            outputs=[quality])

    model_estimator.get_layer('src_embedding').trainable = False
    model_estimator.get_layer('ref_embedding').trainable = False
    model_estimator.get_layer('encoder').trainable = False
    model_estimator.get_layer('decoder').trainable = False
    model_estimator.get_layer('td_attention_state').trainable = False
    model_estimator.get_layer('td_t_tilda').trainable = False
    model_estimator.get_layer('td_t_out').trainable = False

    model_estimator.compile(
            optimizer="adagrad",
            loss={
                "quality": "mse"
            },
            metrics={
                "quality": ["mse", "mae"]
            }
        )

    _printModelSummary(model_estimator, "model_estimator")

    return model_multitask, model_predictor, model_estimator


def train_model(workspaceDir, modelName, devFileSuffix,
                batchSize, epochs, vocab_size, training_mode,
                **kwargs):
    logger.info("initializing TQE training")
    fileBasename = os.path.join(workspaceDir, "tqe." + modelName)

    srcVocabTransformer = WordIndexTransformer(vocab_size=vocab_size)
    refVocabTransformer = WordIndexTransformer(vocab_size=vocab_size)

    X_train, y_train, X_dev, y_dev = _prepareInput(
                                        fileBasename,
                                        srcVocabTransformer,
                                        refVocabTransformer,
                                        devFileSuffix=devFileSuffix,
                                        )

    model_multitask, model_predictor, model_estimator = \
        getModel(srcVocabTransformer, refVocabTransformer, **kwargs)

    logger.info("Training")
    if training_mode == "multitask":
        logger.info("Using multitask training")
        model_multitask.fit([
                X_train['src'],
                X_train['ref']
            ], [
                X_train['ref'].reshape((len(X_train['ref']), -1, 1)),
                y_train
            ],
            batch_size=batchSize,
            epochs=epochs,
            validation_data=([
                    X_dev['src'],
                    X_dev['mt']
                ], [
                    X_dev['ref'].reshape((len(X_dev['ref']), -1, 1)),
                    y_dev
                ]
            ),
            verbose=2
        )
    elif training_mode == "two-step":
        logger.info("Using two-step training")
        model_predictor.fit([
                X_train['src'],
                X_train['ref']
            ], [
                X_train['ref'].reshape((len(X_train['ref']), -1, 1)),
            ],
            batch_size=batchSize,
            epochs=epochs,
            validation_data=([
                    X_dev['src'],
                    X_dev['mt']
                ], [
                    X_dev['ref'].reshape((len(X_dev['ref']), -1, 1)),
                ]
            ),
            verbose=2
        )
        model_estimator.fit([
                X_train['src'],
                X_train['ref']
            ], [
                y_train
            ],
            batch_size=batchSize,
            epochs=epochs,
            validation_data=([
                    X_dev['src'],
                    X_dev['mt']
                ], [
                    y_dev
                ]
            ),
            verbose=2
        )
    else:
        raise ValueError("Training mode not recognized")

    # logger.info("Saving model")
    # model.save(fileBasename + "neural.model.h5")

    logger.info("Evaluating")
    utils.evaluate(model_estimator.predict([
        X_dev['src'],
        X_dev['mt']
    ]).reshape((-1,)), y_dev)


def setupArgparse(parser):
    def run(args):
        train_model(args.workspace_dir,
                    args.model_name,
                    devFileSuffix=args.dev_file_suffix,
                    batchSize=args.batch_size,
                    epochs=args.epochs,
                    vocab_size=args.vocab_size,
                    embedding_size=args.embedding_size,
                    gru_size=args.gru_size,
                    qualvec_size=args.qualvec_size,
                    maxout_size=args.maxout_size,
                    maxout_units=args.maxout_units,
                    training_mode="two-step" if args.two_step else "multitask",
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
    parser.add_argument('-v', '--vocab-size', type=int, default=40000,
                        help='Maximum vocab size')
    parser.add_argument('--maxout-units', type=int, default=2,
                        help='Number of maxout units')
    parser.add_argument('--two-step', action="store_true", default=False,
                        help='Use two step training instead of multitask')
    parser.set_defaults(func=run)
