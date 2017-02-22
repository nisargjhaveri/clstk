from objective import Objective


class AggregateObjective(Objective):
    def __init__(self):
        self._objectives = []

    def addObjective(self, weight, objective):
        self._objectives.append((weight, objective))

    def getObjective(self, summary, corpus):
        objectives = map(
            lambda o: (o[0], o[1].getObjective(summary, corpus)),
            self._objectives
        )

        def objective(sentence):
            return sum(map(lambda o: o[0] * o[1](sentence), objectives))
