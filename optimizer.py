from sentence import Sentence
from summary import Summary


class Optimizer(object):
    def __init__(self):
        pass

    def greedy(self, sizeBudget, objective, corpus):
        summary = Summary()
        sentencesLeft = corpus.getSentences()

        while summary.size() < sizeBudget and len(sentencesLeft) > 0:
            objectiveValues = map(objective.getObjective(summary, corpus),
                                  sentencesLeft)
            maxObjectiveValue = max(objectiveValues)

            candidates = [sentencesLeft[i]
                          for i, v in enumerate(objectiveValues)
                          if v == maxObjectiveValue]

            candidateSizes = map(Sentence.size, candidates)
            minSize = min(candidateSizes)

            if summary.size() + minSize > sizeBudget:
                break

            selectedCandidate = candidates[candidateSizes.index(minSize)]

            summary.addSentence(selectedCandidate)
            sentencesLeft.remove(selectedCandidate)

        return summary
