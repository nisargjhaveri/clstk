"""
Integration with external ROUGE tool-kit.

We recommend the use of
https://github.com/nisargjhaveri/ROUGE-1.5.5-unicode

``ROUGE_HOME`` variable needs to be set to run this.
"""

import os
import subprocess
import tempfile


class ExternalRougeScore(object):
    """
    Integration with external ROUGE tool-kit.
    """

    def runRougeExternal(self, configFileName):
        rougeHome = os.getenv("ROUGE_HOME", ".")
        rougeExecutable = os.path.join(rougeHome, "ROUGE-1.5.5.pl")

        command = ([rougeExecutable] +
                   "-n 2 -m -x -z SPL".split() +  # -f B
                   [configFileName])

        rougeOutput = subprocess.Popen(command)
        rougeOutput.communicate()

    def rouge(self, summaryRefsList):
        """
        Runs external ROUGE-1.5.5 and prints results

        :param summaryRefsList: List containing pairs of path to summary and
                                list of paths to reference summaries
        """
        configFile = tempfile.NamedTemporaryFile(mode='w', suffix=".lst",
                                                 delete=False)

        for summaryPath, refPaths in summaryRefsList:
            rougeEvalrule = [summaryPath] + refPaths
            configFile.write(" ".join(rougeEvalrule) + "\n")

        configFile.close()

        self.runRougeExternal(configFile.name)

        os.unlink(configFile.name)
