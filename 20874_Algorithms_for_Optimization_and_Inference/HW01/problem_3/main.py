import numpy as np
from PIL import Image
import itertools

TARGET_OBJ = 20871.86  # best objective value found for 100 init with k = 8

img = Image.open("../problem_1/20col.png")
a = np.array(img)
h, w, c = a.shape

def get_tentative_clusters(X, TARGET_OBJ):

    unique_colors, colors_count = np.unique(x.reshape(-1, 3), axis=0, return_counts=True)
    unique_colors = unique_colors / 255

    centroids = []

    subset_msk = itertools.product([0, 1], repeat=len(unique_colors))

    for i, msk in enumerate(subset_msk):
        
        if i % 1e5 == 0:
            print(f'Iteration: {i} / {2**len(unique_colors):09,}')
        
        if sum(msk) == 0:
            continue
        
        subset = unique_colors[np.array(msk) == 1]
        subset_counts = colors_count[np.array(msk) == 1]
        subset_obj = np.sum(np.linalg.norm(subset - subset.mean(axis=0), axis=1) * subset_counts)
        
        if TARGET_OBJ > subset_obj:
            centroids.append(subset.mean(axis=0))
    
    return centroids