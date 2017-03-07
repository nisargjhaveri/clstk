import sys

from summarizer import summarize

if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit("Usage: %s <input_directory>" % sys.argv[0])

    inDir = sys.argv[1]

    summary = summarize(inDir)
    print summary.getFormattedSummary()
