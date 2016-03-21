from unittest import TestCase
from ClusterTree import ClusterTree


class TestClusterTree(TestCase):
    def test_depth(self):
        ct = ClusterTree(indices=range(16), min_leaf_size=1)
        self.assertEqual(ct.depth(), 4)
