import numpy as np
from PIL import Image
import itertools

TARGET_OBJ = 20871.86  # best objective value found for 100 init with k = 8

print('Reading image...')
img = Image.open("./20col.png")
X = np.array(img)
h, w, c = X.shape

unique_colors, colors_count = np.unique(X.reshape(-1, 3), axis=0, return_counts=True)
unique_colors = unique_colors / 255

centroids = []

subset_msk = itertools.product([0, 1], repeat=len(unique_colors))
print('Generating centroids...')
for i, msk in enumerate(subset_msk):
    
    if i % 1e5 == 0:
        print(f'Iteration: {i:,} / {2**len(unique_colors):09,}')
    
    if sum(msk) == 0:
        continue
    
    subset = unique_colors[np.array(msk) == 1]
    subset_counts = colors_count[np.array(msk) == 1]
    subset_obj = np.sum(np.linalg.norm(subset - subset.mean(axis=0), axis=1) * subset_counts)
    
    if TARGET_OBJ > subset_obj:
        centroids.append(subset.mean(axis=0))

centroids = centroids[:100]
with open('centroids.txt', 'w') as f:
    for centroid in centroids:
        f.write(f'{centroid[0]},{centroid[1]},{centroid[2]}\n')

print('Writing main.dat...')
path = 'main.dat'
k = 8
file = open(path, 'w')

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

# Max number of centroids
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
file.close()