from typing import TypeVar, Generic, List, Tuple
import random

import numpy as np
from bitarray import bitarray
import bitarray.util as ba

from . import logger

def insert(arr: List[np.ndarray], idx, d):
    arr[idx] = np.append(arr[idx], d.reshape(1, 256), axis=0)

class KMedian01:
    k: int

    def __init__(self, k: int):
        self.k = k

    def distance(self, a: np.ndarray, b: np.ndarray) -> int:
        return (bitarray(list(a)) ^ bitarray(list(b))).count()

    def median(self, data: np.ndarray) -> np.ndarray:
        num, _L = data.shape
        if num == 0:
            return None
        elif num == 1:
            return data[0]
        else:
            accum = np.sum(data.astype(np.int32), axis=0)

        thresh = (num // 2) + (num % 2)
        return accum >= thresh

    def update(self, data: np.ndarray, centroids: List[np.ndarray]) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        """
        @fn update()
        @brief Update centroids according to input data. 
        """
        _num, L = data.shape
        clusters = [np.empty((0, L))] * self.k
        for d in data:
#            min_dist = np.nan
#            cluster_id = 0
#            for i, c in enumerate(centroids):
#                dist = self.distance(d, c)
#                if not (min_dist <= dist):
#                    min_dist = dist
#                    cluster_id = i
#            clusters[cluster_id] = np.append(clusters[cluster_id], d.reshape(1, 256), axis=0)

            cluster_id = np.apply_along_axis(lambda cent: self.distance(d, cent), axis=1, arr=centroids).argmin()
            clusters[cluster_id] = np.append(clusters[cluster_id], d.reshape(1, 256), axis=0)

#        np.apply_along_axis(lambda desc: insert(clusters, np.apply_along_axis(lambda cent: self.distance(desc, cent), axis=1, arr=centroids).argmin(), desc), axis=1, arr=data)

        new_centroids = np.empty_like(centroids)
        for i, cluster in enumerate(clusters):
            new_centroids[i] = self.median(cluster)

        return (new_centroids, clusters)

    def run(self, data: np.ndarray) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        logger.log(self, f"Num of Data: {data.shape[0]}")
        centroids = self.gen_seeds(data)
        clusters = []
        converge = False
        cnt = 0
        while not converge:
            #logger.log(self, f"Cycle #{cnt+1}")
            new_centroids, clusters = self.update(data, centroids)
#            for new, old in zip(new_centroids, centroids):
#                print(f"Update \n{ba.ba2hex(bitarray(list(old)))} => \n{ba.ba2hex(bitarray(list(new)))}")
            #if (new_centroids == centroids).all(): converge = True
            converge = (new_centroids == centroids).all()
            centroids = new_centroids
            cnt += 1

        logger.log(self, f"{cnt} Cycles. ")
        return (centroids, clusters)
            

    def gen_seeds(self, data: np.ndarray) -> np.ndarray:
        #logger.log(self, f"Generating seed points...")
        _num, L = data.shape
        seeds = np.empty((0, L))

        # Choose first seed
        seeds = np.append(seeds, random.choice(data).reshape(1, L), axis=0)
        #logger.log(self, f"Seed #{len(seeds)}: {ba.ba2hex(bitarray(list(seeds[-1])))}")

        # Calc weights and choose the other seeds
        for _ in range(self.k - 1):
            weights = []
            for d in data:
#                min_dist = np.nan
#                for s in seeds:
#                    dist = self.distance(s, d)
#                    if not (min_dist <= dist):
#                        min_dist = dist

                min_dist = np.apply_along_axis(lambda s: self.distance(d, s), axis=1, arr=seeds).min()
                weights.append(min_dist ** 2)

            seeds = np.append(seeds, random.choices(data, weights=weights)[0].reshape(1, L), axis=0)
            #logger.log(self, f"Seed #{len(seeds)}: {ba.ba2hex(bitarray(list(seeds[-1])))}")

        return seeds