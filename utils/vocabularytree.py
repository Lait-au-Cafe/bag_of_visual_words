from typing import Tuple, List

import numpy as np
from bitarray import bitarray
import bitarray.util as ba

from . import kmedian
from . import logger
from .core import bd_distance

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

    @property
    def leaves(self) -> List["Tree"]:
        # leaf
        if len(self.children) == 0:
            return [self]

        # not leaf
        leaves = []
        for child in self.children:
            leaves.extend(child.leaves)
        return leaves

    def reach(self, single_data) -> "Tree":
        # If this is leaf node, return its centroid as representative. 
        if len(self.children) == 0: return self
        
        # Otherwise, proceed to the next level. 
        next_child_id = np.array(list(map(lambda child: bd_distance(child.centroid, single_data), self.children))).argmin()
        return self.children[next_child_id].reach(single_data)


class VacabularyTree:
    kw: int
    lw: int
    tree: Tree

    def __init__(self, kw: int, lw: int):
        self.kw = kw
        self.lw = lw

    @property
    def words(self):
        return np.array([leaf.centroid for leaf in self.tree.leaves])

    def represent(self, desc):
        return self.tree.reach(desc).centroid

    def build(self, frame_features):
        # Extract descriptors from features. 
        descriptors = np.empty((0, 256))
        for _kps, descs in frame_features:
            descriptors = np.append(descriptors, descs, axis=0)

        #==============================
        # Build up tree
        #==============================
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

        self.tree.display()

        # Create sets of words per frame. 
        N = len(frame_features)
        logger.log(self, f"Creatig frame word sets. ")
        frame_word_sets = []
        for _kps, descs in frame_features:
            words = np.apply_along_axis(lambda desc: ba.ba2hex(bitarray(list(self.represent(desc)))), axis=1, arr=descs)
            frame_word_sets.append(set(words))

        #==============================
        # Calc weights at words
        #==============================
        # Calc idf = ln(N/Ni)
        # tf part depends on source image, so it's not calculated here. 
        for leaf in self.tree.leaves:
            word = ba.ba2hex(bitarray(list(leaf.centroid)))
            Ni = len([1 for word_set in frame_word_sets if word in word_set])
            #logger.log(self, f"Ni = {Ni}")
            leaf.weight = np.log(N / Ni)
    
    def calc_bowv(self, features: np.ndarray) -> np.ndarray:
        return np.array([])

    def save(self):
        # Hey someone, please implement this part. 
        pass