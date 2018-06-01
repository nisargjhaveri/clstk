import sys


class ProgressBar(object):
    """
    Class to manage and show pretty progress-bar in the console
    """

    def __init__(self, totalCount):
        """
        Initialize the progressbar

        :param totalCount: Total items to be processed
        """
        self._totalCount = totalCount
        self._pastProgress = 0

        print ("----5---10---15---20---25---30---35---40---45---50" +
               "---55---60---65---70---75---80---85---90---95--100")

    def _printProgress(self, progress, newline=False):
        sys.stdout.write("*" * progress)

        if newline:
            sys.stdout.write("\n")

        sys.stdout.flush()

    def done(self, doneCount):
        """
        Move progressbar ahead

        :param doneCount: Out of ``totalCount``, this many have been processed
        """
        progressTo = int(doneCount * 100 / self._totalCount)

        self._printProgress(progressTo - self._pastProgress)

        self._pastProgress = progressTo

    def complete(self):
        """
        Complete progress
        """
        self._printProgress(100 - self._pastProgress, newline=True)
        self._pastProgress = 100
