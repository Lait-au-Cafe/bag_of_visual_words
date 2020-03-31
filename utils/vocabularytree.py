from typing import Tuple, List

import numpy as np
from bitarray import bitarray
import bitarray.util as ba

from . import kmedian
from . import logger

class Tree:
    children: List['Tree']
    weight: float = None
    centroid: np.ndarray

    def __init__(self, centroid):
        self.centroid = centroid
        self.children = []

    def add_child(self, child: "Tree"):
        self.children.append(child)

    def display(self, hid = 0, tid = 1, level = 0):
        cent_str = f"{ba.ba2hex(bitarray(list(self.centroid)))}" if self.centroid is not None else "Root"
        print("\t" * level + f"[{hid}] {cent_str} [{tid}]")
        for i, child in enumerate(self.children):
            child.display(tid, tid + (i + 1), level + 1)

class VacabularyTree:
    kw: int
    lw: int
    tree: Tree

    def __init__(self, kw: int, lw: int):
        self.kw = kw
        self.lw = lw

    def build(self, frame_features):
        # Extract descriptors from features. 
        descriptors = np.empty((0, 256))
        for _kps, descs in frame_features:
            descriptors = np.append(descriptors, descs, axis=0)

        # Build up tree
        k_median = kmedian.KMedian01(self.kw)
        clusters = [descriptors]
        self.tree = Tree(None) # top Llevel
        tree_ptrs = [self.tree]
        for level in range(self.lw):
            logger.log(self, f"Processing Level {level}...")
            new_clusters = []
            new_tree_ptrs = []
            for tree_ptr, cluster in zip(tree_ptrs, clusters):
                for cent, clus in zip(*k_median.run(cluster)):
                    new_clusters.append(clus)
                    child = Tree(cent)
                    tree_ptr.add_child(child)
                    new_tree_ptrs.append(child)
            clusters = new_clusters
            tree_ptrs = new_tree_ptrs

            import sys
            reclim = sys.getrecursionlimit()
            sys.setrecursionlimit(10**6)
            self.tree.display()
            sys.setrecursionlimit(reclim)

        # Calc weights at words