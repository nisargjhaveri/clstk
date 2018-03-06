import os
import sys
import tempfile
import subprocess
import argparse

from summarizer.utils import fs
from summarizer.utils import nlp

from summarizer.evaluation import RougeScore

from summarizer import linBilmes
from summarizer import coRank
from summarizer import simFusion


def runSummarizer(inDir, outFile, summarizer, args):
    args = argparse.Namespace(**vars(args))
    args.source_directory = inDir

    summary = summarizer(args, silent=True)
    with open(outFile, "w") as f:
        f.write(summary.getTargetSummary().encode('utf8'))


def summarizeAll(docNames, docsDir, outDir, summarizer, args):
    fs.ensureDir(outDir)

    total = len(docNames)

    for i, inDirName in enumerate(docNames):
        inDir = os.path.join(docsDir, inDirName)
        outFile = os.path.join(outDir, inDirName)

        print "Summarizing:", i + 1, "/", total, "\r",
        sys.stdout.flush()
        runSummarizer(inDir, outFile, summarizer, args)

    print


def runRougeExternal(configFileName):
    rougeHome = os.getenv("ROUGE_HOME", ".")
    rougeExecutable = os.path.join(rougeHome, "ROUGE-1.5.5.pl")

    command = ([rougeExecutable] +
               "-n 2 -m -x -z SPL".split() +  # -f B
               [configFileName])

    rougeOutput = subprocess.Popen(command)
    rougeOutput.communicate()


def getAvailableReferences(refsDir):
    return os.walk(refsDir).next()[1]


def getRougeScore(summaryNames, summariesDir, refsDir):
    configFile = tempfile.NamedTemporaryFile(mode='w', suffix=".lst",
                                             delete=False)
    summaryRefsList = []

    for summaryName in summaryNames:
        summaryPath = os.path.join(summariesDir, summaryName)
        summaryRefsDir = os.path.join(refsDir, summaryName)
        refPaths = map(lambda f: os.path.join(summaryRefsDir, f),
                       os.walk(summaryRefsDir).next()[2])

        rougeEvalrule = [summaryPath] + refPaths

        summaryRefsList.append((summaryPath, refPaths))

        configFile.write(" ".join(rougeEvalrule) + "\n")

    configFile.close()

    runRougeExternal(configFile.name)
    print "-"
    RougeScore(stemmer=nlp.getStemmer()).rouge(summaryRefsList)

    os.unlink(configFile.name)


if __name__ == '__main__':
    common_parser = argparse.ArgumentParser(add_help=False)

    common_parser.add_argument('source_path',
                               help='Directory containing all the source '
                               'files to be summarized. Each set of documents '
                               'are expected to be in different directories '
                               'inside this path.')
    common_parser.add_argument('models_path',
                               help='Directory containing all the model '
                               'summaries. Each set of summaires are expected '
                               'to be in different directory inside this '
                               'path, having the same name as the '
                               'corresponding directory in the source '
                               'directory.')
    common_parser.add_argument('summaries_path',
                               help='Directory to store the generated '
                               'summaries. The directory will be created if '
                               'not already exists.')
    common_parser.add_argument('--only-rouge', action='store_true',
                               help='Do not run summarizer. '
                               'Only compule ROUGE score for existing '
                               'summaries in summaries_path')

    parser = argparse.ArgumentParser(
            description='Evaluate the summarizer',
            epilog='Set ROUGE_HOME environment variable for this to work')

    subparsers = parser.add_subparsers(title='methods',
                                       description='Summarization method')

    linBilmes.setupArgparse(subparsers.add_parser('linBilmes',
                                                  parents=[common_parser]))
    coRank.setupArgparse(subparsers.add_parser('coRank',
                                               parents=[common_parser]))
    simFusion.setupArgparse(subparsers.add_parser('simFusion',
                                                  parents=[common_parser]))

    args = parser.parse_args()

    docNames = getAvailableReferences(args.models_path)

    if not args.only_rouge:
        summarizeAll(docNames, args.source_path, args.summaries_path,
                     args.func, args)

    getRougeScore(docNames, args.summaries_path, args.models_path)
