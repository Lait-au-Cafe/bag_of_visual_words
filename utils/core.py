from typing import Dict

import numpy as np
import bitarray as ba
#import bitarray.util as ba

# Calc the distance between two binary descriptor. 
def bd_distance(a: np.ndarray, b: np.ndarray) -> int:
    return (ba.bitarray(list(a)) ^ ba.bitarray(list(b))).count()

# Calc the norm of bag-of-words vector. 
def bowv_norm(v: Dict[int, float]) -> float:
    # use L1 norm according to the paper. 
    return sum(map(abs, v.values()))

# Normalize bag-of-words vector given an argument. 
# Returns norm. 
def bowv_normalize(v: Dict[int, float]) -> float:
    norm = bowv_norm(v)
    if norm != 0: 
        for k in v.keys(): v[k] /= norm
    return norm
    

# Calc the similarity between two bag-of-words vector. 
# Similarity is [0..1], 0: most different, 1: most similar. 
def bowv_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 0