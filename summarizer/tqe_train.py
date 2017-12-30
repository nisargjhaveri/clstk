import argparse
import logging

from summarizer.tqe import train

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
    parser.add_argument('--evaluate', action='store_true',
                        help='Also evaluate the trained model.')
    parser.add_argument('--eval-file-suffix', type=str, default=None,
                        help='Suffix for test files')
    parser.add_argument('--feature-file-suffix', type=str, default=None,
                        help='Suffix for feature files')
    parser.add_argument('--lm-exists', action='store_true',
                        help='Don\'t train language model. It already exists.')
    parser.add_argument('--ngrams-exists', action='store_true',
                        help='Don\'t compute ngram freqs. It already exists.')
    parser.add_argument('--parse-exists', action='store_true',
                        help='Don\'t parse sentences. It already exists.')

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

    train.train_model(args.workspace_dir,
                      args.model_name,
                      evaluate=args.evaluate,
                      evalFileSuffix=args.eval_file_suffix,
                      featureFileSuffix=args.feature_file_suffix,
                      trainLM=(not args.lm_exists),
                      trainNGrams=(not args.ngrams_exists),
                      parseSentences=(not args.parse_exists))
