from ._coverageObjective import CoverageObjective
from ._dievrsityRewardObjective import DiversityRewardObjective
from ._translationQualityObjective import TranslationQualityObjective


def getObjectives():
    return [
        ('coverage', 1.0, CoverageObjective),
        ('diversity', 6.0, DiversityRewardObjective),
        ('tqe', 0.0, TranslationQualityObjective),
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
