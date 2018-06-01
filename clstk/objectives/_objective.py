class Objective(object):
    def setCorpus(self, corpus):
        raise NotImplementedError("To be implemented by all subclasses")

    def getObjective(self, summary, corpus):
        # Should return a function that takes a sentence and
        # returns objective value
        raise NotImplementedError("To be implemented by all subclasses")

    @staticmethod
    def getParams():
        # Should return list of .utils.Param
        return []
