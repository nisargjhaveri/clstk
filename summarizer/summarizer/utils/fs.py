import os


def ensureDir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
