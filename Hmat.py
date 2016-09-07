"""HMat.py"""


class Hmat(object):
    """Implement a hierarchical Matrix"""
    def __init__(self):
        self.tl = None
        self.tr = None
        self.bl = None
        self.br = None


class RMat(object):
    """Rank-k matrix"""
