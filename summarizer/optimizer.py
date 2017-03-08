from sentence import Sentence
from summary import Summary


class Optimizer(object):
    def __init__(self):
        pass

    def greedy(self, sizeBudget, objective, corpus):
        summary = Summary()
        sentencesLeft = corpus.getSentences()

        objective.setCorpus(corpus)

        while summary.size() < sizeBudget and len(sentencesLeft) > 0:
            objectiveValues = map(objective.getObjective(summary),
                                  sentencesLeft)
            maxObjectiveValue = max(objectiveValues)

            candidates = [sentencesLeft[i]
                          for i, v in enumerate(objectiveValues)
                          if v == maxObjectiveValue]

            candidateSizes = map(Sentence.size, candidates)
            minSize = min(candidateSizes)

            selectedCandidate = candidates[candidateSizes.index(minSize)]
            sentencesLeft.remove(selectedCandidate)

            if summary.size() + minSize <= sizeBudget:
                summary.addSentence(selectedCandidate)

            budgetLeft = sizeBudget - summary.size()
            sentencesLeft = filter(lambda s: s.size() < budgetLeft,
                                   sentencesLeft)

        return summary
