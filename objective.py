class Objective(object):
    def getObjective(self, summary, corpus):
        # Should return a function that takes a sentence and
        # returns objective value
        raise NotImplementedError("To be implemented by all subclasses")
