import argparse
import logging

from summarizer.tqe import baseline

from summarizer.utils import colors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Train Translation Quality Estimation')
    parser.add_argument('--no-colors', action='store_true',
                        help='Don\'t show colors in verbose log')
    parser.add_argument('workspace_dir',
                        help='Directory containing prepared files')
    parser.add_argument('model_name',
                        help='Identifier for prepared files used with ' +
                        'preparation')
    parser.add_argument('--dev-file-suffix', type=str, default=None,
                        help='Suffix for test files')
    parser.add_argument('--feature-file-suffix', type=str, default=None,
                        help='Suffix for feature files')
    parser.add_argument('--train-lm', action='store_true',
                        help='Don\'t train language model. It already exists.')
    parser.add_argument('--train-ngrams', action='store_true',
                        help='Don\'t compute ngram freqs. It already exists.')
    parser.add_argument('--parse', action='store_true',
                        help='Don\'t parse sentences. It already exists.')
    parser.add_argument('--normalize', action='store_true',
                        help='Weather to normalize features or not.')
    parser.add_argument('--tune', action='store_true',
                        help='Weather to normalize features or not.')
    parser.add_argument('--max-jobs', type=int, default=-1,
                        help='Maximum number of jobs to run parallelly')

    args = parser.parse_args()

    if args.no_colors:
        colors.disable()

    logLevel = logging.NOTSET
    logging.basicConfig(
        level=logLevel,
        format=(colors.enclose('%(asctime)s', colors.CYAN) +
                colors.enclose('.%(msecs)03d ', colors.BLUE) +
                colors.enclose('%(name)s', colors.YELLOW) + ': ' +
                '%(message)s'),
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    baseline.train_model(args.workspace_dir,
                         args.model_name,
                         devFileSuffix=args.dev_file_suffix,
                         featureFileSuffix=args.feature_file_suffix,
                         normalize=args.normalize,
                         tune=args.tune,
                         trainLM=args.train_lm,
                         trainNGrams=args.train_ngrams,
                         parseSentences=args.parse,
                         maxJobs=args.max_jobs)