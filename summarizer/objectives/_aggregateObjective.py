from ._objective import Objective

import logging
logger = logging.getLogger("aggregateObjective.py")


class AggregateObjective(Objective):
    def __init__(self):
        self._objectives = []

    def addObjective(self, weight, objective):
        logger.info("Adding objective `%s` with weight: %f",
                    objective.__class__.__name__, weight)
        self._objectives.append((weight, objective))

    def setCorpus(self, corpus):
        for weight, objective in self._objectives:
            objective.setCorpus(corpus)

    def getObjective(self, summary):
        objectives = map(
            lambda o: (o[0], o[1].getObjective(summary)),
            self._objectives
        )

        def objective(sentence):
            return sum(map(lambda o: o[0] * o[1](sentence), objectives))

        return objective
