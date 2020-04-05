import numpy as np
import bitarray as ba
#import bitarray.util as ba

# Calc the distance between two binary descriptor. 
def bd_distance(a: np.ndarray, b: np.ndarray) -> int:
    return (ba.bitarray(list(a)) ^ ba.bitarray(list(b))).count()

# Calc the norm of bag-of-words vector. 
def bowv_norm(a: np.ndarray) -> float:
    # use L1 norm according to the paper. 
    # Note: every element of bow vector cannot be a negative value. 
    return a.sum()


# Calc the similarity between two bag-of-words vector. 
# Similarity is [0..1], 0: most different, 1: most similar. 
def bowv_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 0