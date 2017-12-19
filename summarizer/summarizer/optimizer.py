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

        sizeBudget, countTokens = sizeBudget
        sizeName = "tokens" if countTokens else "chars"

        def sentenceSize(sent):
            return sent.tokenCount() if countTokens else sent.charCount()

        def summarySize(summary):
            return summary.tokenCount() if countTokens else summary.charCount()

        logger.info("Greedily optimizing the objective")
        logger.info("Summary budget: %d %s", sizeBudget, sizeName)
        while summarySize(summary) < sizeBudget and len(sentencesLeft) > 0:
            objectiveValues = map(objective.getObjective(summary),
                                  sentencesLeft)
            maxObjectiveValue = max(objectiveValues)

            candidates = [sentencesLeft[i]
                          for i, v in enumerate(objectiveValues)
                          if v == maxObjectiveValue]

            candidateSizes = map(sentenceSize, candidates)
            minSize = min(candidateSizes)

            selectedCandidate = candidates[candidateSizes.index(minSize)]
            sentencesLeft.remove(selectedCandidate)

            if summarySize(summary) + minSize <= sizeBudget:
                logger.info("Sentence added with objective value: %f, " +
                            "size: %d", maxObjectiveValue, minSize)
                summary.addSentence(selectedCandidate)

            budgetLeft = sizeBudget - summarySize(summary)
            sentencesLeft = filter(lambda s: sentenceSize(s) < budgetLeft,
                                   sentencesLeft)

        logger.info("Optimization done, summary size: %d chars, %d tokens",
                    summary.charCount(), summary.tokenCount())

        return summary
