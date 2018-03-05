import argparse
import os
import shelve


def setupSubparsers(parser):
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('workspace_dir',
                               help='Directory containing prepared files')
    common_parser.add_argument('--save-model', type=str, default=None,
                               help='Save the model with this basename')
    common_parser.add_argument('data_name',
                               help='Identifier for prepared files')
    common_parser.add_argument('--dev-file-suffix', type=str, default=None,
                               help='Suffix for dev files')
    common_parser.add_argument('--test-file-suffix', type=str, default=None,
                               help='Suffix for test files')

    subparsers = parser.add_subparsers(title='models', dest="model",
                                       description='TQE model to train')

    parent = {'parents': [common_parser]}
    baselineArgparser(subparsers.add_parser('baseline', **parent))
    postechArgparser(subparsers.add_parser('postech', **parent))
    rnnArgparser(subparsers.add_parser('rnn', **parent))


def getModel(model):
    if model == "baseline":
        from summarizer.tqe import baseline
        return baseline
    elif model == "postech":
        from summarizer.tqe import postech
        return postech
    elif model == "rnn":
        from summarizer.tqe import rnn
        return rnn


def train(args):
    if args.save_model:
        shelf = shelve.open(os.path.join(args.workspace_dir,
                                         "model." + args.save_model))

        shelf['args'] = args

        shelf.close()

    getModel(args.model).train(args)


def baselineArgparser(parser):
    parser.add_argument('--feature-file-suffix', type=str, default=None,
                        help='Suffix for feature files')
    parser.add_argument('--train-lm', action='store_true',
                        help='Train language model.')
    parser.add_argument('--train-ngrams', action='store_true',
                        help='Compute ngram freqs.')
    parser.add_argument('--parse', action='store_true',
                        help='Parse sentences.')
    parser.add_argument('--normalize', action='store_true',
                        help='Weather to normalize features or not.')
    parser.add_argument('--tune', action='store_true',
                        help='Weather to tune parameters or not.')
    parser.add_argument('--max-jobs', type=int, default=-1,
                        help='Maximum number of jobs to run parallelly')


def postechArgparser(parser):
    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=25,
                        help='Number of epochs to run')
    parser.add_argument('--ensemble-count', type=int, default=3,
                        help='Number of models to ensemble')
    parser.add_argument('--max-len', type=int, default=100,
                        help='Maximum length of the sentences')
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
    parser.add_argument('--predictor-model', type=str, default=None,
                        help='Name of predictor model to save/load from')
    parser.add_argument('--two-step', action="store_true",
                        help='Use two step training instead of multitask')
    parser.add_argument('--predictor-data', type=str, default=None,
                        help='Identifier for prepared data to train predictor')


def rnnArgparser(parser):
    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=25,
                        help='Number of epochs to run')
    parser.add_argument('--ensemble-count', type=int, default=3,
                        help='Number of models to ensemble')
    parser.add_argument('--max-len', type=int, default=100,
                        help='Maximum length of the sentences')
    parser.add_argument('--source-embeddings', type=str, default=None,
                        help='fastText model name for target language')
    parser.add_argument('--target-embeddings', type=str, default=None,
                        help='fastText model name for target language')
    parser.add_argument('-m', '--embedding-size', type=int, default=300,
                        help='Size of word embeddings')
    parser.add_argument('-n', '--gru-size', type=int, default=500,
                        help='Size of GRU')
    parser.add_argument('-v', '--vocab-size', type=int, default=40000,
                        help='Maximum vocab size')
    parser.add_argument('--with-attention', action="store_true",
                        help='Add attention in decoder')
    parser.add_argument('--summary-attention', action="store_true",
                        help='Get quality summary using attention')
    parser.add_argument('--no-estimator', action="store_true",
                        help='Don\'t use separate estimator layer')
