import numpy as np
from PIL import Image

MAX_OBJ = 4814760.748061607  # best objective value found for 100 init with k = 8

def obj(X, y):
    """
    Computes the sum of the L2 distance between a set of vectors X of shape
    (n_vectors, n_features) and a target vector y of shape (n_features,)  
    
    """
    return np.sum(np.sum((X - y)**2, axis=1)**0.5)



def get_tentative_clusters(X, MAX_OBJ, obj):
    clusters = []
    n = len(X)
    s_list = list(X)  # for indexing

    for i in range(2 ** n):
        cluster = []
        for j in range(n):
            # Check if the j-th element should be included
            if (i & (1 << j)):
                cluster.append(s_list[j])
        if obj(np.array(cluster), np.mean(cluster, axis=0)) < MAX_OBJ:
            clusters.append(cluster) 

