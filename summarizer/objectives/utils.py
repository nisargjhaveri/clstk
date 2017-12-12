from ._coverageObjective import CoverageObjective
from ._dievrsityRewardObjective import DiversityRewardObjective


def getObjectives():
    return [
        ('coverage', 1.0, CoverageObjective),
        ('diversity', 6.0, DiversityRewardObjective)
    ]


def addObjectiveParams(parser):
    for key, weight, objective in getObjectives():
        parser.add_argument(
            '--{}-lambda'.format(key), default=weight,
            type=float, metavar="lambda"
        )

        for param in objective.getParams():
            param.addParamToParser(parser, key)


def getParams(args):
    args = vars(args)
    params = {}

    for key, _, objective in getObjectives():
        params[key] = {}
        params[key]['lambda'] = args['{}_lambda'.format(key)]

        for param in objective.getParams():
            params[key][param.getName()] = \
                        args['{}_{}'.format(key, param.getName())]

    return params


class Param(object):
    def __init__(self, name, type, default, metavar, help):
        self.name = name
        self.type = type
        self.default = default
        self.metavar = metavar
        self.help = help

    def addParamToParser(self, parser, key):
        parser.add_argument(
            '--{}-{}'.format(key, self.name),
            type=self.type,
            default=self.default,
            metavar=self.metavar,
            help=self.help
        )

    def getName(self):
        return self.name
