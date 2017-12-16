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
