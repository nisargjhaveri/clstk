import os
import shelve

from . import utils

from keras.layers import Layer, multiply, concatenate, average
from keras.layers import Input, Embedding, Dense, Reshape
from keras.layers import RNN, GRU, GRUCell, Bidirectional
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

import keras.backend as K

from keras.preprocessing.sequence import pad_sequences
from keras.utils.generic_utils import CustomObjectScope

from .common import WordIndexTransformer
from .common import _loadSentences, _loadData, _preprocessSentences
from .common import _printModelSummary, TimeDistributedSequential
from .common import pearsonr


import logging
logger = logging.getLogger("postech")


def _loadPredictorData(fileBasename):
    predictorData = {
        "src": [],
        "ref": []
    }

    if fileBasename:
        srcSentencesPath = fileBasename + ".src"
        refSentencesPath = fileBasename + ".ref"

        predictorData['src'] = _loadSentences(srcSentencesPath)
        predictorData['ref'] = _loadSentences(refSentencesPath)

    return predictorData


def _prepareInput(workspaceDir, modelName,
                  srcVocabTransformer, refVocabTransformer,
                  max_len,
                  devFileSuffix=None, testFileSuffix=None,
                  predictorDataModel=None):
    logger.info("Loading data")

    X_train, y_train, X_dev, y_dev, X_test, y_test = _loadData(
                    os.path.join(workspaceDir, "tqe." + modelName),
                    devFileSuffix, testFileSuffix
                )

    pred_train = _loadPredictorData(
                    os.path.join(workspaceDir, "tqe." + predictorDataModel)
                    if predictorDataModel else None
                )

    logger.info("Transforming sentences to onehot")

    srcVocabTransformer \
        .fit(X_train['src']) \
        .fit(X_dev['src']) \
        .fit(X_test['src']) \
        .fit(pred_train['src'])

    srcSentencesTrain = srcVocabTransformer.transform(X_train['src'])
    srcSentencesDev = srcVocabTransformer.transform(X_dev['src'])
    srcSentencesTest = srcVocabTransformer.transform(X_test['src'])
    srcPredictorTrain = srcVocabTransformer.transform(pred_train['src'])

    refVocabTransformer.fit(X_train['mt']) \
                       .fit(X_dev['mt']) \
                       .fit(X_test['mt']) \
                       .fit(X_train['ref']) \
                       .fit(X_dev['ref']) \
                       .fit(X_test['ref']) \
                       .fit(pred_train['ref'])

    mtSentencesTrain = refVocabTransformer.transform(X_train['mt'])
    mtSentencesDev = refVocabTransformer.transform(X_dev['mt'])
    mtSentencesTest = refVocabTransformer.transform(X_test['mt'])
    refSentencesTrain = refVocabTransformer.transform(X_train['ref'])
    refSentencesDev = refVocabTransformer.transform(X_dev['ref'])
    refSentencesTest = refVocabTransformer.transform(X_test['ref'])
    refPredictorTrain = refVocabTransformer.transform(pred_train['ref'])

    def getMaxLen(listOfsequences):
        return max([max(map(len, sequences)) for sequences in listOfsequences
                    if len(sequences)])

    srcMaxLen = min(getMaxLen([srcSentencesTrain, srcSentencesDev,
                               srcPredictorTrain]), max_len)
    refMaxLen = min(getMaxLen([mtSentencesTrain, mtSentencesDev,
                               refSentencesTrain, refSentencesDev,
                               refPredictorTrain]), max_len)

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

    X_test = {
        "src": pad_sequences(srcSentencesTest, maxlen=srcMaxLen),
        "mt": pad_sequences(mtSentencesTest, maxlen=refMaxLen),
        "ref": pad_sequences(refSentencesTest, maxlen=refMaxLen)
    }

    pred_train = {
        "src": pad_sequences(srcPredictorTrain, maxlen=srcMaxLen),
        "ref": pad_sequences(refPredictorTrain, maxlen=refMaxLen),
    } if predictorDataModel else None

    return X_train, y_train, X_dev, y_dev, X_test, y_test, pred_train


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

    def compute_mask(self, inputs, mask=None):
        return mask[-1]

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


class ConcatDecoder(Layer):
    def call(self, inputs):
        return K.concatenate(inputs)

    def compute_mask(self, inputs, mask=None):
        return mask[0][0]

    def compute_output_shape(self, input_shape):
        out_shape = list(input_shape[0])
        out_shape[-1] += input_shape[1][-1]
        return tuple(out_shape)


def getModel(srcVocabTransformer, refVocabTransformer,
             embedding_size,
             gru_size,
             qualvec_size,
             maxout_size,
             maxout_units,
             model_inputs=None, verbose=False,
             ):
    src_vocab_size = srcVocabTransformer.vocab_size()
    ref_vocab_size = refVocabTransformer.vocab_size()

    if verbose:
        logger.info("Creating model")

    if model_inputs:
        src_input, ref_input = model_inputs
    else:
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

    qualvec_pre = multiply([out_state, W_y], name="pre_qevf")  # Pre-QEVF
    qualvec_post = ConcatDecoder(name="post_qevf")(
                        decoder
                    )  # Post-QEVF
    # Lambda(
    #     K.concatenate,
    #     output_shape=lambda x: (x[0][0], x[0][1], x[0][2] + x[1][2]),
    #     name="post_qevf"
    # )(decoder)  # Post-QEVF

    qualvec = concatenate([qualvec_pre, qualvec_post], name="qualvec")

    quality_summary = Bidirectional(GRU(gru_size), name="estimator")(qualvec)

    quality = Dense(1, name="quality")(quality_summary)

    model_multitask = Model(inputs=[src_input, ref_input],
                            outputs=[predicted_word, quality])
    model_multitask.compile(
            optimizer="adadelta",
            loss={
                "predicted_word": "sparse_categorical_crossentropy",
                "quality": "mse"
            },
            metrics={
                "predicted_word": ["sparse_categorical_accuracy"],
                "quality": ["mse", "mae", pearsonr]
            }
        )
    if verbose:
        _printModelSummary(logger, model_multitask, "model_multitask")

    model_predictor = Model(inputs=[src_input, ref_input],
                            outputs=[predicted_word])
    model_predictor.compile(
            optimizer="adadelta",
            loss={
                "predicted_word": "sparse_categorical_crossentropy",
            },
            metrics={
                "predicted_word": ["sparse_categorical_accuracy"],
            }
        )
    if verbose:
        _printModelSummary(logger, model_predictor, "model_predictor")

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
            optimizer="adadelta",
            loss={
                "quality": "mse"
            },
            metrics={
                "quality": ["mse", "mae", pearsonr]
            }
        )
    if verbose:
        _printModelSummary(logger, model_estimator, "model_estimator")

    return model_multitask, model_predictor, model_estimator


def getEnsembledModel(ensemble_count, **kwargs):
    if ensemble_count == 1:
        return getModel(verbose=True, **kwargs)

    src_input = Input(shape=(None, ))
    ref_input = Input(shape=(None, ))

    model_inputs = [src_input, ref_input]

    logger.info("Creating models to ensemble")
    verbose = [True] + [False] * (ensemble_count - 1)
    models = [getModel(model_inputs=model_inputs, verbose=v, **kwargs)
              for _, v in zip(range(ensemble_count), verbose)]

    models_multitask, models_predictor, models_estimator = zip(*models)

    multitask_outputs = [model([src_input, ref_input])
                         for model in models_multitask]

    predicted_words, qualities = zip(*multitask_outputs)
    predicted_word = average(list(predicted_words), name='predicted_word')
    quality = average(list(qualities), name='quality')

    predictor_output = average([model([src_input, ref_input])
                                for model in models_predictor],
                               name='predicted_word')

    estimator_output = average([model([src_input, ref_input])
                                for model in models_estimator],
                               name='quality')

    logger.info("Compiling ensembled models")
    model_multitask = Model(inputs=[src_input, ref_input],
                            outputs=[predicted_word, quality])
    model_multitask.compile(
            optimizer="adadelta",
            loss={
                "predicted_word": "sparse_categorical_crossentropy",
                "quality": "mse"
            },
            metrics={
                "predicted_word": ["sparse_categorical_accuracy"],
                "quality": ["mse", "mae", pearsonr]
            }
        )

    model_predictor = Model(inputs=[src_input, ref_input],
                            outputs=predictor_output)
    model_predictor.compile(
            optimizer="adadelta",
            loss={
                "predicted_word": "sparse_categorical_crossentropy",
            },
            metrics={
                "predicted_word": ["sparse_categorical_accuracy"],
            }
        )

    model_estimator = Model(inputs=[src_input, ref_input],
                            outputs=estimator_output)
    model_estimator.compile(
            optimizer="adadelta",
            loss={
                "quality": "mse"
            },
            metrics={
                "quality": ["mse", "mae", pearsonr]
            }
        )

    return model_multitask, model_predictor, model_estimator


def train_model(workspaceDir, modelName, devFileSuffix, testFileSuffix,
                saveModel,
                batchSize, epochs, max_len, vocab_size, training_mode,
                predictor_model, predictor_data,
                **kwargs):
    logger.info("initializing TQE training")

    predictorModelFile = None
    if predictor_model:
        predictorModelFile = os.path.join(
                        workspaceDir,
                        ".".join(["tqe", predictor_model, "predictor.model"])
                    )

    srcVocabTransformer = WordIndexTransformer(vocab_size=vocab_size)
    refVocabTransformer = WordIndexTransformer(vocab_size=vocab_size)

    X_train, y_train, X_dev, y_dev, X_test, y_test, pred_train = _prepareInput(
                                        workspaceDir,
                                        modelName,
                                        srcVocabTransformer,
                                        refVocabTransformer,
                                        max_len=max_len,
                                        devFileSuffix=devFileSuffix,
                                        testFileSuffix=testFileSuffix,
                                        predictorDataModel=predictor_data
                                        )

    model_multitask, model_predictor, model_estimator = \
        getEnsembledModel(srcVocabTransformer=srcVocabTransformer,
                          refVocabTransformer=refVocabTransformer,
                          **kwargs)

    if predictorModelFile and not pred_train:
        logger.info("Loading weights for predictor")
        model_predictor.load_weights(predictorModelFile)

    logger.info("Training")
    if training_mode not in ["multitask", "two-step"]:
        raise ValueError("Training mode not recognized")

    if pred_train:
        logger.info("Training predictor on predictor data")

        callbacks = None
        if predictorModelFile:
            callbacks = [ModelCheckpoint(
                            filepath=(predictorModelFile + ".{epoch:02d}"),
                            save_weights_only=True)]

        model_predictor.fit([
                pred_train['src'],
                pred_train['ref']
            ], [
                pred_train['ref'].reshape((len(pred_train['ref']), -1, 1)),
            ],
            batch_size=batchSize,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks
        )
        if predictorModelFile:
            logger.info("Saving weights for predictor")
            model_predictor.save_weights(predictorModelFile)

    if training_mode == "multitask":
        logger.info("Training multitask model")
        model_multitask.fit([
                X_train['src'],
                X_train['mt']
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
            callbacks=[
                EarlyStopping(monitor="val_quality_pearsonr", patience=2,
                              mode="max"),
            ],
            verbose=2
        )

    if training_mode == "two-step":
        logger.info("Training predictor")
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
        logger.info("Training estimator")
        model_estimator.fit([
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
    if saveModel:
        logger.info("Saving model")
        shelf = shelve.open(os.path.join(workspaceDir, "model." + saveModel))

        models = [model_multitask, model_predictor, model_estimator]

        shelf['config'] = [model.get_config() for model in models]
        shelf['weights'] = [model.get_weights() for model in models]
        shelf['params'] = {
            'srcVocabTransformer': srcVocabTransformer,
            'refVocabTransformer': refVocabTransformer,
        }

        shelf.close()

    logger.info("Evaluating on development data of size %d" % len(y_dev))
    utils.evaluate(model_estimator.predict([
        X_dev['src'],
        X_dev['mt']
    ]).reshape((-1,)), y_dev)

    logger.info("Evaluating on test data of size %d" % len(y_test))
    utils.evaluate(model_estimator.predict([
        X_test['src'],
        X_test['mt']
    ]).reshape((-1,)), y_test)


def load_predictor(workspaceDir, saveModel, max_len, **kwargs):
    shelf = shelve.open(os.path.join(workspaceDir, "model." + saveModel), 'r')

    srcVocabTransformer = shelf['params']['srcVocabTransformer']
    refVocabTransformer = shelf['params']['refVocabTransformer']

    model_multitask, model_predictor, model_estimator = \
        getEnsembledModel(srcVocabTransformer=srcVocabTransformer,
                          refVocabTransformer=refVocabTransformer,
                          **kwargs)

    models = [model_multitask, model_predictor, model_estimator]

    logger.info("Loading weights into models")
    for model, weights in zip(models, shelf['weights']):
        model.set_weights(weights)

    shelf.close()

    def predictor(src, mt):
        if not isinstance(src, list):
            src = [src]
            mt = [mt]

        src = _preprocessSentences(src)
        mt = _preprocessSentences(mt)

        src = srcVocabTransformer.transform(src)
        mt = refVocabTransformer.transform(mt)

        srcMaxLen = min(max(map(len, src)), max_len)
        refMaxLen = min(max(map(len, mt)), max_len)

        src = pad_sequences(src, maxlen=srcMaxLen)
        mt = pad_sequences(mt, maxlen=refMaxLen)

        return model_estimator.predict([src, mt]).reshape((-1,))

    return predictor


def _get_training_mode(args):
    if args.two_step:
        return "two-step"
    else:
        return "multitask"


def train(args):
    train_model(args.workspace_dir,
                args.data_name,
                devFileSuffix=args.dev_file_suffix,
                testFileSuffix=args.test_file_suffix,
                saveModel=args.save_model,
                batchSize=args.batch_size,
                epochs=args.epochs,
                ensemble_count=args.ensemble_count,
                vocab_size=args.vocab_size,
                max_len=args.max_len,
                embedding_size=args.embedding_size,
                gru_size=args.gru_size,
                qualvec_size=args.qualvec_size,
                maxout_size=args.maxout_size,
                maxout_units=args.maxout_units,
                training_mode=_get_training_mode(args),
                predictor_model=args.predictor_model,
                predictor_data=args.predictor_data,
                )


def getPredictor(args):
    return load_predictor(args.workspace_dir,
                          saveModel=args.save_model,
                          ensemble_count=args.ensemble_count,
                          max_len=args.max_len,
                          embedding_size=args.embedding_size,
                          gru_size=args.gru_size,
                          qualvec_size=args.qualvec_size,
                          maxout_size=args.maxout_size,
                          maxout_units=args.maxout_units,
                          )
