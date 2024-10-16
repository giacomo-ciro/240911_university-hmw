import numpy as np
from PIL import Image
import itertools
import sys

if len(sys.argv) > 1:
    TARGET_OBJ = float(sys.argv[1])
else:
    TARGET_OBJ = 4347298.25  # best objective value found for 1000 init with k = 8 unormalized with frequency+ and noise N(0, 15)

print('Reading image...')
img = Image.open("./20col.png")
X = np.array(img)
h, w, c = X.shape
unique_colors, colors_count = np.unique(X.reshape(-1, 3), axis=0, return_counts=True)

N = unique_colors.shape[0]
k = 8

# -------------------------------------- #
# ----- Generate centroids -------------- #
# -------------------------------------- #
print('Generating centroids...')
centroids = []
subset_msk = itertools.product([0, 1], repeat=len(unique_colors))
for i, msk in enumerate(subset_msk):
    
    if i % 1e5 == 0:
        print(f'Iteration: {i:,} / {2**len(unique_colors):09,}')
    
    subset_cardinality = sum(msk)
    if subset_cardinality == 0 or subset_cardinality >= (N-k+2):
        continue
    
    subset = unique_colors[np.array(msk) == 1]
    subset_counts = colors_count[np.array(msk) == 1]
    subset_obj = np.sum(np.linalg.norm(subset - subset.mean(axis=0), axis=1) * subset_counts)

    # If already above target, skip
    if TARGET_OBJ < subset_obj:
        continue

    centroids.append(subset.mean(axis=0))
    
print(f'Centroids found: {len(centroids):,}')

centroids = np.array(centroids)
idx = np.random.choice(np.arange(centroids.shape[0]), 2000, replace=False)
centroids = centroids[idx]

# Save to reconstruct image later
np.save('./assets/centroids.npy', centroids)
np.save('./assets/unique_colors.npy', unique_colors)

# -------------------------------------- #
# ----- Write main.dat ----------------- #
# -------------------------------------- #
print('Writing main.dat...')
path = 'main.dat'
k = 8

with open(path, 'w') as file:
    
    # Colors
    file.write('set I :=')
    for j in range(len(unique_colors)):
        file.write(f' {j+1}')
    file.write(';\n\n')

    # Centroids
    file.write('set J :=')
    for i in range(len(centroids)):
        file.write(f' {i+1}')
    file.write(';\n\n')

    # Max number of centroids
    file.write(f'param k := {k};\n\n')

    # Num colors
    file.write(f'param N := {len(unique_colors)};\n\n')

    # Distance of color i to centroid j
    file.write('param d :')
    for j in range(len(centroids)):
        file.write(f' {j+1}')

    file.write(' :=\n')
    for i in range(len(unique_colors)):
        file.write(f'{i+1}')
        for j in range(len(centroids)):
            file.write(f' {np.linalg.norm(centroids[j] - unique_colors[i]):.2f}')
        if i!= len(unique_colors)-1:
            file.write('\n')
    file.write(';\n\n')

    # Number of pixels of color i
    file.write('param w :=\n')
    for i in range(len(unique_colors)):
        file.write(f' {i+1} {colors_count[i]}')
        if i!= len(unique_colors)-1:
            file.write('\n')
    file.write(';')
print('Done!')