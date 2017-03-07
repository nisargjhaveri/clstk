import argparse

from summarizer import summarize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Automatically summarize a set of documents')
    parser.add_argument('source_directory',
                        help='Directory containing a set of files to be ' +
                        'summarized.')

    args = parser.parse_args()

    summary = summarize(args.source_directory)
    print summary.getFormattedSummary()
