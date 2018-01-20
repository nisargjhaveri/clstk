import sys


class ProgressBar(object):
    def __init__(self, totalCount):
        self._totalCount = totalCount
        self._pastProgress = 0

        print ("----5---10---15---20---25---30---35---40---45---50" +
               "---55---60---65---70---75---80---85---90---95--100")

    def _printProgress(self, progress):
        sys.stdout.write("*" * progress)
        sys.stdout.flush()

    def done(self, doneCount):
        progressTo = int(doneCount * 100 / self._totalCount)

        self._printProgress(progressTo - self._pastProgress)

        self._pastProgress = progressTo

    def complete(self):
        self._printProgress(100 - self._pastProgress)
        self._pastProgress = 100

        # Print newline at the end
        print
