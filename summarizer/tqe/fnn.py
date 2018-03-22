import os
import shelve

from . import utils

from keras.layers import average
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.callbacks import EarlyStopping

from .common import _printModelSummary
from .common import pearsonr


from .baseline import _loadAndPrepareFeatures


import logging
logger = logging.getLogger("fnn")


def getModel(model_inputs=None, num_features=None, verbose=False):
    if verbose:
        logger.info("Creating model")

    if model_inputs:
        feature_input, = model_inputs
    else:
        feature_input = Input(shape=(num_features, ))

    hidden = feature_input
    # hidden = Dense(100)(feature_input)
    # hidden = Dense(100)(hidden)
    # hidden = Dense(100)(hidden)
    # hidden = Dense(100)(hidden)
    # hidden = Dense(100)(hidden)
    quality = Dense(1, name="quality")(hidden)

    if verbose:
        logger.info("Compiling model")
    model = Model(inputs=feature_input,
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


def getEnsembledModel(ensemble_count, num_features, **kwargs):
    if ensemble_count == 1:
        return getModel(num_features=num_features, verbose=True, **kwargs)

    feature_input = Input(shape=(num_features, ))

    model_inputs = [feature_input]

    logger.info("Creating models to ensemble")
    verbose = [True] + [False] * (ensemble_count - 1)
    models = [getModel(model_inputs=model_inputs, verbose=v, **kwargs)
              for v in verbose]

    output = average([model(feature_input) for model in models],
                     name='quality')

    logger.info("Compiling ensembled model")
    model = Model(inputs=feature_input,
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


def train_model(workspaceDir, modelName,
                devFileSuffix, testFileSuffix,
                featureFileSuffix, normalize, trainLM, trainNGrams,
                saveModel, batchSize, epochs,
                **kwargs):
    logger.info("initializing TQE training")
    fileBasename = os.path.join(workspaceDir, "tqe." + modelName)

    X_train, y_train, X_dev, y_dev, X_test, y_test = _loadAndPrepareFeatures(
        fileBasename,
        devFileSuffix=devFileSuffix, testFileSuffix=testFileSuffix,
        featureFileSuffix=featureFileSuffix,
        normalize=normalize,
        trainLM=trainLM,
        trainNGrams=trainNGrams,
    )

    num_features = len(X_train[0])

    model = getEnsembledModel(num_features=num_features, **kwargs)

    logger.info("Training model")
    model.fit([
            X_train
        ], [
            y_train
        ],
        batch_size=batchSize,
        epochs=epochs,
        # shuffle=True,
        validation_data=([
                X_dev
            ], [
                y_dev
            ]
        ),
        callbacks=[
            # EarlyStopping(monitor="val_pearsonr", patience=2, mode="max"),
        ],
        verbose=1
    )

    if saveModel:
        logger.info("Saving model")
        shelf = shelve.open(os.path.join(workspaceDir, "model." + saveModel))

        shelf['config'] = model.get_config()
        shelf['weights'] = model.get_weights()
        shelf['params'] = {}

        shelf.close()

    logger.info("Evaluating on development data of size %d" % len(y_dev))
    utils.evaluate(model.predict([
            X_dev
        ]).reshape((-1,)), y_dev)

    logger.info("Evaluating on test data of size %d" % len(y_test))
    utils.evaluate(model.predict([
            X_test
        ]).reshape((-1,)), y_test)


def train(args):
    train_model(args.workspace_dir,
                args.data_name,
                devFileSuffix=args.dev_file_suffix,
                testFileSuffix=args.test_file_suffix,

                saveModel=args.save_model,
                batchSize=args.batch_size,
                epochs=args.epochs,
                ensemble_count=args.ensemble_count,

                featureFileSuffix=args.feature_file_suffix,
                normalize=args.normalize,
                trainLM=args.train_lm,
                trainNGrams=args.train_ngrams,)
