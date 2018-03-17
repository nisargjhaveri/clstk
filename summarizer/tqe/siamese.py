import os
import shelve

from . import utils

from keras.layers import dot, average, concatenate
from keras.layers import Input, Embedding
from keras.layers import Dense, Activation, Lambda
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout
from keras.layers import GRU, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping

import keras.backend as K

# from keras.utils.generic_utils import CustomObjectScope

from .common import WordIndexTransformer, _loadData
from .common import _printModelSummary, TimeDistributedSequential
from .common import pad_sequences, getBatchGenerator
from .common import pearsonr
from .common import get_fastText_embeddings


import logging
logger = logging.getLogger("siamese")


def _prepareInput(workspaceDir, modelName,
                  srcVocabTransformer, refVocabTransformer,
                  max_len, num_buckets,
                  devFileSuffix=None, testFileSuffix=None,
                  ):
    logger.info("Loading data")

    X_train, y_train, X_dev, y_dev, X_test, y_test = _loadData(
                    os.path.join(workspaceDir, "tqe." + modelName),
                    devFileSuffix, testFileSuffix
                )

    logger.info("Transforming sentences to onehot")

    srcVocabTransformer \
        .fit(X_train['src']) \
        .fit(X_dev['src']) \
        .fit(X_test['src'])

    srcSentencesTrain = srcVocabTransformer.transform(X_train['src'])
    srcSentencesDev = srcVocabTransformer.transform(X_dev['src'])
    srcSentencesTest = srcVocabTransformer.transform(X_test['src'])

    refVocabTransformer.fit(X_train['mt']) \
                       .fit(X_dev['mt']) \
                       .fit(X_test['mt']) \
                       .fit(X_train['ref']) \
                       .fit(X_dev['ref']) \
                       .fit(X_test['ref'])

    mtSentencesTrain = refVocabTransformer.transform(X_train['mt'])
    mtSentencesDev = refVocabTransformer.transform(X_dev['mt'])
    mtSentencesTest = refVocabTransformer.transform(X_test['mt'])
    refSentencesTrain = refVocabTransformer.transform(X_train['ref'])
    refSentencesDev = refVocabTransformer.transform(X_dev['ref'])
    refSentencesTest = refVocabTransformer.transform(X_test['ref'])

    def getMaxLen(listOfsequences):
        return max([max(map(len, sequences)) for sequences in listOfsequences
                    if len(sequences)])

    srcMaxLen = min(getMaxLen([srcSentencesTrain, srcSentencesDev]), max_len)
    refMaxLen = min(getMaxLen([mtSentencesTrain, mtSentencesDev,
                               refSentencesTrain, refSentencesDev]), max_len)

    pad_args = {'num_buckets': num_buckets}
    X_train = {
        "src": pad_sequences(srcSentencesTrain, maxlen=srcMaxLen, **pad_args),
        "mt": pad_sequences(mtSentencesTrain, maxlen=refMaxLen, **pad_args),
        "ref": pad_sequences(refSentencesTrain, maxlen=refMaxLen, **pad_args)
    }

    X_dev = {
        "src": pad_sequences(srcSentencesDev, maxlen=srcMaxLen, **pad_args),
        "mt": pad_sequences(mtSentencesDev, maxlen=refMaxLen, **pad_args),
        "ref": pad_sequences(refSentencesDev, maxlen=refMaxLen, **pad_args)
    }

    X_test = {
        "src": pad_sequences(srcSentencesTest, maxlen=srcMaxLen, **pad_args),
        "mt": pad_sequences(mtSentencesTest, maxlen=refMaxLen, **pad_args),
        "ref": pad_sequences(refSentencesTest, maxlen=refMaxLen, **pad_args)
    }

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def getSentenceEncoder(vocabTransformer,
                       embedding_size, gru_size,
                       fastText,
                       attention,
                       use_cnn,
                       filter_sizes, num_filters, sentence_vector_size,
                       cnn_dropout,
                       model_inputs, verbose,
                       ):
    vocab_size = vocabTransformer.vocab_size()

    embedding_kwargs = {}

    if fastText:
        if verbose:
            logger.info("Loading fastText embeddings from: " + fastText)
        embedding_kwargs['weights'] = [get_fastText_embeddings(
                                fastText,
                                vocabTransformer,
                                embedding_size
                                )]

    if model_inputs:
        input, = model_inputs
    else:
        input = Input(shape=(None, ))

    embedding = Embedding(
                        output_dim=embedding_size,
                        input_dim=vocab_size,
                        mask_zero=(not use_cnn),
                        name="embedding",
                        **embedding_kwargs)(input)

    if use_cnn:
        conv_blocks = []
        for filter_size in filter_sizes:
            conv = Conv1D(
                        filters=num_filters,
                        kernel_size=filter_size
                    )(embedding)
            conv = GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)

        z = concatenate(conv_blocks) \
            if len(conv_blocks) > 1 else conv_blocks[0]

        if cnn_dropout > 0:
            z = Dropout(cnn_dropout)(z)

        encoder = Dense(sentence_vector_size)(z)
    else:
        encoder = Bidirectional(
                        GRU(gru_size, return_sequences=attention),
                        name="encoder"
                )(embedding)

        if attention:
            attention_weights = TimeDistributedSequential([
                Dense(gru_size, activation="tanh"),
                Dense(1, name="attention_weights"),
            ], encoder)

            # attention_weights = Reshape((-1,))(attention_weights)
            attention_weights = Lambda(
                        lambda x: K.reshape(x, (x.shape[0], -1,)),
                        output_shape=lambda input_shape: input_shape[:-1],
                        mask=lambda inputs, mask: mask,
                        name="reshape"
                        )(attention_weights)

            attention_weights = Activation(
                                    "softmax",
                                    name="attention_softmax"
                                )(attention_weights)

            encoder = dot([attention_weights, encoder],
                          axes=(1, 1),
                          name="summary"
                          )

    sentence_encoder = Model(inputs=input, outputs=encoder)

    if verbose:
        _printModelSummary(logger, sentence_encoder, "sentence_encoder")

    return sentence_encoder


def getModel(srcVocabTransformer, refVocabTransformer,
             src_fastText, ref_fastText,
             model_inputs=None, verbose=False,
             **kwargs
             ):
    if verbose:
        logger.info("Creating model")

    if model_inputs:
        src_input, ref_input = model_inputs
    else:
        src_input = Input(shape=(None, ))
        ref_input = Input(shape=(None, ))

    src_sentence_enc = getSentenceEncoder(vocabTransformer=srcVocabTransformer,
                                          fastText=src_fastText,
                                          model_inputs=[src_input],
                                          verbose=verbose,
                                          **kwargs)(src_input)

    ref_sentence_enc = getSentenceEncoder(vocabTransformer=refVocabTransformer,
                                          fastText=ref_fastText,
                                          model_inputs=[ref_input],
                                          verbose=verbose,
                                          **kwargs)(ref_input)

    quality = dot([src_sentence_enc, ref_sentence_enc],
                  axes=-1,
                  normalize=True,
                  name="quality")

    if verbose:
        logger.info("Compiling model")
    model = Model(inputs=[src_input, ref_input],
                  outputs=[quality])

    model.compile(
            optimizer="adadelta",
            loss={
                "quality": "mse"
            },
            metrics={
                "quality": ["mse", "mae", pearsonr]
            }
        )
    if verbose:
        _printModelSummary(logger, model, "model")

    return model


def getEnsembledModel(ensemble_count, **kwargs):
    if ensemble_count == 1:
        return getModel(verbose=True, **kwargs)

    src_input = Input(shape=(None, ))
    ref_input = Input(shape=(None, ))

    model_inputs = [src_input, ref_input]

    logger.info("Creating models to ensemble")
    verbose = [True] + [False] * (ensemble_count - 1)
    models = [getModel(model_inputs=model_inputs, verbose=v, **kwargs)
              for v in verbose]

    output = average([model([src_input, ref_input]) for model in models],
                     name='quality')

    logger.info("Compiling ensembled model")
    model = Model(inputs=[src_input, ref_input],
                  outputs=output)

    model.compile(
            optimizer="adadelta",
            loss={
                "quality": "mse"
            },
            metrics={
                "quality": ["mse", "mae", pearsonr]
            }
        )
    _printModelSummary(logger, model, "ensembled_model")

    return model


def train_model(workspaceDir, modelName, devFileSuffix, testFileSuffix,
                saveModel,
                batchSize, epochs, max_len, num_buckets, vocab_size,
                **kwargs):
    logger.info("initializing TQE training")

    srcVocabTransformer = WordIndexTransformer(vocab_size=vocab_size)
    refVocabTransformer = WordIndexTransformer(vocab_size=vocab_size)

    X_train, y_train, X_dev, y_dev, X_test, y_test = _prepareInput(
                                        workspaceDir,
                                        modelName,
                                        srcVocabTransformer,
                                        refVocabTransformer,
                                        max_len=max_len,
                                        num_buckets=num_buckets,
                                        devFileSuffix=devFileSuffix,
                                        testFileSuffix=testFileSuffix,
                                        )

    def get_embedding_path(model):
        return os.path.join(workspaceDir,
                            "fastText",
                            ".".join([model, "bin"])
                            ) if model else None

    kwargs['src_fastText'] = get_embedding_path(kwargs['src_fastText'])
    kwargs['ref_fastText'] = get_embedding_path(kwargs['ref_fastText'])

    model = getEnsembledModel(srcVocabTransformer=srcVocabTransformer,
                              refVocabTransformer=refVocabTransformer,
                              **kwargs)

    logger.info("Training model")

    model.fit_generator(getBatchGenerator([
                X_train['src'],
                X_train['mt']
            ], [
                y_train
            ],
            key=lambda x: "_".join(map(str, map(len, x))),
            batch_size=batchSize
        ),
        epochs=epochs,
        shuffle=True,
        validation_data=getBatchGenerator([
                X_dev['src'],
                X_dev['mt']
            ], [
                y_dev
            ],
            key=lambda x: "_".join(map(str, map(len, x)))
        ),
        callbacks=[
            EarlyStopping(monitor="val_pearsonr", patience=2, mode="max"),
        ],
        verbose=2
    )

    if saveModel:
        logger.info("Saving model")
        shelf = shelve.open(os.path.join(workspaceDir, "model." + saveModel))

        shelf['config'] = model.get_config()
        shelf['weights'] = model.get_weights()
        shelf['params'] = {
            'srcVocabTransformer': srcVocabTransformer,
            'refVocabTransformer': refVocabTransformer,
        }

        shelf.close()

    logger.info("Evaluating on development data of size %d" % len(y_dev))
    dev_batches = getBatchGenerator([
            X_dev['src'],
            X_dev['mt']
        ],
        key=lambda x: "_".join(map(str, map(len, x)))
    )
    y_dev = dev_batches.align(y_dev)
    utils.evaluate(model.predict_generator(dev_batches).reshape((-1,)),
                   y_dev)

    logger.info("Evaluating on test data of size %d" % len(y_test))
    test_batches = getBatchGenerator([
            X_test['src'],
            X_test['mt']
        ],
        key=lambda x: "_".join(map(str, map(len, x)))
    )
    y_test = test_batches.align(y_test)
    utils.evaluate(model.predict_generator(test_batches).reshape((-1,)),
                   y_test)


def load_predictor(workspaceDir, saveModel, max_len, num_buckets, **kwargs):
    raise NotImplementedError()


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
                num_buckets=args.buckets,
                embedding_size=args.embedding_size,
                src_fastText=args.source_embeddings,
                ref_fastText=args.target_embeddings,
                gru_size=args.gru_size,
                attention=args.with_attention,
                use_cnn=args.cnn,
                filter_sizes=args.filter_sizes,
                num_filters=args.num_filters,
                sentence_vector_size=args.sentence_vector_size,
                cnn_dropout=args.cnn_dropout,
                )


def getPredictor(args):
    raise NotImplementedError()
