from sentence import Sentence
from summary import Summary

import logging
logger = logging.getLogger("optimizer.py")


class Optimizer(object):
    def __init__(self):
        pass

    def greedy(self, sizeBudget, objective, corpus):
        summary = Summary()
        sentencesLeft = corpus.getSentences()

        objective.setCorpus(corpus)

        logger.info("Greedily optimizing the objective")
        logger.info("Summary budget: %d", sizeBudget)
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
                logger.info("Sentence added with objective value: %f, " +
                            "size: %d", maxObjectiveValue, minSize)
                summary.addSentence(selectedCandidate)

            budgetLeft = sizeBudget - summary.size()
            sentencesLeft = filter(lambda s: s.size() < budgetLeft,
                                   sentencesLeft)

        logger.info("Optimization done, summary size: %d", summary.size())

        return summary
