import argparse
import os
import shelve

from summarizer.tqe import baseline
from summarizer.tqe import postech
from summarizer.tqe import rnn


def setupSubparsers(subparsers):
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

    parent = {'parents': [common_parser]}
    baseline.setupArgparse(subparsers.add_parser('baseline', **parent))
    postech.setupArgparse(subparsers.add_parser('postech', **parent))
    rnn.setupArgparse(subparsers.add_parser('rnn', **parent))


def run(args):
    if args.save_model:
        shelf = shelve.open(os.path.join(args.workspace_dir,
                                         "model." + args.save_model))

        copy_args = argparse.Namespace(**vars(args))
        copy_args.func = None
        shelf['args'] = copy_args

        shelf.close()

    args.func(args)
