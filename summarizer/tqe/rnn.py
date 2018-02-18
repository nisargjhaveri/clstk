import os

from . import utils

from keras.layers import Input, Embedding, Dense
from keras.layers import GRU, GRUCell, Bidirectional
from keras.models import Model

import keras.backend as K

from keras.preprocessing.sequence import pad_sequences

from .common import WordIndexTransformer, _loadData
from .common import _printModelSummary
from .common import pearsonr


import logging
logger = logging.getLogger("rnn")


def _prepareInput(workspaceDir, modelName,
                  srcVocabTransformer, refVocabTransformer,
                  max_len,
                  devFileSuffix=None, predictorDataModel=None):
    logger.info("Loading data")

    X_train, y_train, X_dev, y_dev = _loadData(
                    os.path.join(workspaceDir, "tqe." + modelName),
                    devFileSuffix
                )

    logger.info("Transforming sentences to onehot")

    srcVocabTransformer \
        .fit(X_train['src']) \
        .fit(X_dev['src'])

    srcSentencesTrain = srcVocabTransformer.transform(X_train['src'])
    srcSentencesDev = srcVocabTransformer.transform(X_dev['src'])

    refVocabTransformer.fit(X_train['mt']) \
                       .fit(X_dev['mt']) \
                       .fit(X_train['ref']) \
                       .fit(X_dev['ref'])

    mtSentencesTrain = refVocabTransformer.transform(X_train['mt'])
    mtSentencesDev = refVocabTransformer.transform(X_dev['mt'])
    refSentencesTrain = refVocabTransformer.transform(X_train['ref'])
    refSentencesDev = refVocabTransformer.transform(X_dev['ref'])

    def getMaxLen(listOfsequences):
        return max([max(map(len, sequences)) for sequences in listOfsequences
                    if len(sequences)])

    srcMaxLen = min(getMaxLen([srcSentencesTrain, srcSentencesDev]), max_len)
    refMaxLen = min(getMaxLen([mtSentencesTrain, mtSentencesDev,
                               refSentencesTrain, refSentencesDev]), max_len)

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


def getModel(srcVocabTransformer, refVocabTransformer,
             embedding_size,
             gru_size
             ):
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

    decoder = Bidirectional(
                    GRU(gru_size, return_sequences=True, return_state=True),
                    name="decoder"
            )(
              ref_embedding,
              initial_state=encoder[1:]
            )

    quality_summary = Bidirectional(
                    GRU(gru_size),
                    name="estimator"
            )(decoder[0])

    quality = Dense(1, name="quality")(quality_summary)

    logger.info("Compiling model")
    model = Model(inputs=[src_input, ref_input],
                  outputs=[quality])
    model.compile(
            optimizer="adagrad",
            loss={
                "quality": "mse"
            },
            metrics={
                "quality": ["mse", "mae", pearsonr]
            }
        )
    _printModelSummary(logger, model, "model")

    return model


def train_model(workspaceDir, modelName, devFileSuffix,
                batchSize, epochs, max_len, vocab_size,
                **kwargs):
    logger.info("initializing TQE training")

    srcVocabTransformer = WordIndexTransformer(vocab_size=vocab_size)
    refVocabTransformer = WordIndexTransformer(vocab_size=vocab_size)

    X_train, y_train, X_dev, y_dev = _prepareInput(
                                        workspaceDir,
                                        modelName,
                                        srcVocabTransformer,
                                        refVocabTransformer,
                                        max_len=max_len,
                                        devFileSuffix=devFileSuffix,
                                        )

    model = getModel(srcVocabTransformer, refVocabTransformer, **kwargs)

    logger.info("Training model")
    model.fit([
            X_train['src'],
            X_train['mt']
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

    # logger.info("Saving model")
    # model.save(fileBasename + "neural.model.h5")

    logger.info("Evaluating")
    utils.evaluate(model.predict([
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
                    max_len=args.max_len,
                    embedding_size=args.embedding_size,
                    gru_size=args.gru_size,
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
    parser.add_argument('--max-len', type=int, default=100,
                        help='Maximum length of the sentences')
    parser.add_argument('-m', '--embedding-size', type=int, default=300,
                        help='Size of word embeddings')
    parser.add_argument('-n', '--gru-size', type=int, default=500,
                        help='Size of GRU')
    parser.add_argument('-v', '--vocab-size', type=int, default=40000,
                        help='Maximum vocab size')
    parser.set_defaults(func=run)
