import os
import subprocess
import tempfile


class ExternalRougeScore(object):
    def runRougeExternal(self, configFileName):
        rougeHome = os.getenv("ROUGE_HOME", ".")
        rougeExecutable = os.path.join(rougeHome, "ROUGE-1.5.5.pl")

        command = ([rougeExecutable] +
                   "-n 2 -m -x -z SPL".split() +  # -f B
                   [configFileName])

        rougeOutput = subprocess.Popen(command)
        rougeOutput.communicate()

    def rouge(self, summaryRefsList):
        configFile = tempfile.NamedTemporaryFile(mode='w', suffix=".lst",
                                                 delete=False)

        for summaryPath, refPaths in summaryRefsList:
            rougeEvalrule = [summaryPath] + refPaths
            configFile.write(" ".join(rougeEvalrule) + "\n")

        configFile.close()

        self.runRougeExternal(configFile.name)

        os.unlink(configFile.name)
