import os
import shelve

from . import utils

import numpy as np

from keras.layers import average, concatenate
from keras.layers import Input, Embedding
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping

from .common import WordIndexTransformer, _loadData
from .common import _printModelSummary
from .common import pad_sequences, getBatchGenerator
from .common import pearsonr


from .baseline import _loadAndPrepareFeatures


import logging
logger = logging.getLogger("shef")


def _prepareInput(workspaceDir, modelName,
                  srcVocabTransformer, refVocabTransformer,
                  max_len, num_buckets,
                  devFileSuffix=None, testFileSuffix=None,
                  ):
    logger.info("Loading data")

    X_train, y_train, X_dev, y_dev, X_test, y_test = _loadData(
                    os.path.join(workspaceDir, "tqe." + modelName),
                    devFileSuffix, testFileSuffix,
                    tokenize=False
                )

    for split in (X_train, X_dev, X_test):
        for lang in ('src', 'mt', 'ref'):
            split[lang] = map(lambda s: np.array(list(s)), split[lang])

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

    pad_args = {'num_buckets': 0}
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
                       embedding_size,
                       filter_sizes, num_filters, pool_length,
                       cnn_depth,
                       cnn_dropout,
                       model_inputs, verbose,
                       ):
    vocab_size = vocabTransformer.vocab_size()

    input, = model_inputs

    embedding = Embedding(
                        output_dim=embedding_size,
                        input_dim=vocab_size,
                        name="embedding",)(input)

    conv_blocks = []
    for filter_size in filter_sizes:
        conv = embedding
        for i in range(cnn_depth):
            conv = Conv1D(
                        filters=num_filters,
                        kernel_size=filter_size
                    )(conv)
            conv = MaxPooling1D(pool_size=pool_length)(conv)

        conv = Flatten()(conv)

        conv_blocks.append(conv)

    encoder = concatenate(conv_blocks) \
        if len(conv_blocks) > 1 else conv_blocks[0]

    if cnn_dropout > 0:
        encoder = Dropout(cnn_dropout)(encoder)

    sentence_encoder = Model(inputs=input, outputs=encoder)

    if verbose:
        _printModelSummary(logger, sentence_encoder, "sentence_encoder")

    return sentence_encoder


def getModel(srcVocabTransformer, refVocabTransformer,
             num_features, src_len, ref_len,
             model_inputs=None, verbose=False,
             **kwargs
             ):
    if verbose:
        logger.info("Creating model")

    if not model_inputs:
        model_inputs = [
            Input(shape=(num_features, )),
            Input(shape=(src_len, )),
            Input(shape=(ref_len, )),
        ]

    feature_input, src_input, ref_input = model_inputs

    src_sentence_enc = getSentenceEncoder(vocabTransformer=srcVocabTransformer,
                                          model_inputs=[src_input],
                                          verbose=verbose,
                                          **kwargs)(src_input)

    ref_sentence_enc = getSentenceEncoder(vocabTransformer=refVocabTransformer,
                                          model_inputs=[ref_input],
                                          verbose=verbose,
                                          **kwargs)(ref_input)

    features_enc = Dense(50, activation="tanh")(feature_input)
    features_enc = Dense(50, activation="tanh")(features_enc)

    hidden = concatenate([features_enc, src_sentence_enc, ref_sentence_enc])

    hidden = Dense(50, activation="tanh")(hidden)
    hidden = Dense(50, activation="tanh")(hidden)

    quality = Dense(1, name="quality")(hidden)

    if verbose:
        logger.info("Compiling model")
    model = Model(inputs=model_inputs,
                  outputs=[quality])

    model.compile(
            optimizer="sgd",
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


def getEnsembledModel(ensemble_count,
                      num_features, src_len, ref_len,
                      **kwargs):
    kwargs['num_features'] = num_features
    kwargs['src_len'] = src_len
    kwargs['ref_len'] = ref_len

    if ensemble_count == 1:
        return getModel(verbose=True, **kwargs)

    feature_input = Input(shape=(num_features, ))
    src_input = Input(shape=(src_len, ))
    ref_input = Input(shape=(ref_len, ))

    model_inputs = [feature_input, src_input, ref_input]

    logger.info("Creating models to ensemble")
    verbose = [True] + [False] * (ensemble_count - 1)
    models = [getModel(model_inputs=model_inputs, verbose=v, **kwargs)
              for v in verbose]

    output = average([model(model_inputs) for model in models],
                     name='quality')

    logger.info("Compiling ensembled model")
    model = Model(inputs=model_inputs,
                  outputs=output)

    model.compile(
            optimizer="sgd",
            loss={
                "quality": "mse"
            },
            metrics={
                "quality": ["mse", "mae", pearsonr]
            }
        )
    _printModelSummary(logger, model, "ensembled_model")

    return model


def train_model(workspaceDir, modelName,
                devFileSuffix, testFileSuffix,
                featureFileSuffix, normalize, trainLM, trainNGrams,
                max_len, num_buckets, vocab_size,
                saveModel, batchSize, epochs,
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

    X_train['features'], _, X_dev['features'], _, X_test['features'], _ = \
        _loadAndPrepareFeatures(
            os.path.join(workspaceDir, "tqe." + modelName),
            devFileSuffix=devFileSuffix, testFileSuffix=testFileSuffix,
            featureFileSuffix=featureFileSuffix,
            normalize=normalize,
            trainLM=trainLM,
            trainNGrams=trainNGrams,
        )

    num_features = len(X_train['features'][0])
    src_len = len(X_train['src'][0])
    ref_len = len(X_train['ref'][0])

    model = getEnsembledModel(num_features=num_features,
                              src_len=src_len,
                              ref_len=ref_len,
                              srcVocabTransformer=srcVocabTransformer,
                              refVocabTransformer=refVocabTransformer,
                              **kwargs)

    logger.info("Training model")
    model.fit_generator(getBatchGenerator([
                X_train['features'],
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
                X_dev['features'],
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
            X_dev['features'],
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
            X_test['features'],
            X_test['src'],
            X_test['mt']
        ],
        key=lambda x: "_".join(map(str, map(len, x)))
    )
    y_test = test_batches.align(y_test)
    utils.evaluate(model.predict_generator(test_batches).reshape((-1,)),
                   y_test)


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
                filter_sizes=args.filter_sizes,
                num_filters=args.num_filters,
                pool_length=args.pool_length,
                cnn_depth=args.cnn_depth,
                cnn_dropout=args.cnn_dropout,

                featureFileSuffix=args.feature_file_suffix,
                normalize=args.normalize,
                trainLM=args.train_lm,
                trainNGrams=args.train_ngrams,)
