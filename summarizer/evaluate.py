import os
import tempfile
import subprocess
import argparse

from summarizer import summarize
import utils


def runSummarizer(inDir, outFile):
    summary = summarize(inDir)
    with open(outFile, "w") as f:
        f.write(summary.getFormattedSummary().encode('utf8'))


def summarizeAll(docsDir, outDir):
    dirNames = os.walk(docsDir).next()[1]

    utils.ensureDir(outDir)

    total = len(dirNames)

    for i, inDirName in enumerate(dirNames):
        inDir = os.path.join(docsDir, inDirName)
        outFile = os.path.join(outDir, inDirName)

        print "Summarizing:", i + 1, "/", total, "\r",
        runSummarizer(inDir, outFile)

    print


def runRougeExternal(configFileName):
    rougeHome = os.getenv("ROUGE_HOME", ".")
    rougeExecutable = os.path.join(rougeHome, "ROUGE-1.5.5.pl")

    command = [rougeExecutable] + "-a -n 2 -z SPL".split() + [configFileName]

    rougeOutput = subprocess.Popen(command)
    rougeOutput.communicate()


def getRougeScore(summariesDir, refsDir):
    configFile = tempfile.NamedTemporaryFile(mode='w', suffix=".lst",
                                             delete=False)

    summaryNames = os.walk(summariesDir).next()[2]

    for summaryName in summaryNames:
        summaryPath = os.path.join(summariesDir, summaryName)
        summaryRefsDir = os.path.join(refsDir, summaryName)
        refPaths = map(lambda f: os.path.join(summaryRefsDir, f),
                       os.walk(summaryRefsDir).next()[2])

        rougeEvalrule = [summaryPath] + refPaths

        configFile.write(" ".join(rougeEvalrule) + "\n")

    configFile.close()

    runRougeExternal(configFile.name)

    os.unlink(configFile.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Evaluate the summarizer',
            epilog='Set ROUGE_HOME enviromental variable for this to work')
    parser.add_argument('source_path',
                        help='Directory containing all the source files to ' +
                        'be summarized. Each set of documents are expected ' +
                        'to be in different directories inside this path.')
    parser.add_argument('models_path',
                        help='Directory containing all the model summaries. ' +
                        'Each set of summaires are expected to be in ' +
                        'different directory inside this path, having the ' +
                        'same name as the corresponding directory in the ' +
                        'source directory.')
    parser.add_argument('summaries_path',
                        help='Directory to store the generated summaries. ' +
                        'The directory will be created if not already exists.')
    parser.add_argument('--only-rouge', action='store_true',
                        help='Do not run summarizer. ' +
                        'Only compule ROUGE score for existing ' +
                        'summaries in summaries_path')

    args = parser.parse_args()

    if not args.only_rouge:
        summarizeAll(args.source_path, args.summaries_path)

    getRougeScore(args.summaries_path, args.models_path)
