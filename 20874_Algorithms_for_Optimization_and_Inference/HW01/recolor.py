import sys
import numpy as np
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------------- #
# ----- Get params from cmd ------------ #
# -------------------------------------- #
if len(sys.argv) != 4:
    print('Provide required arguments:\n\tpython recolor.py <input_img> <output_img> <k>')
    sys.exit(1)
input_img = sys.argv[1]     # path to input file
output_img = sys.argv[2]    # path to output file
k = int(sys.argv[3])        # number of colors

# -------------------------------------- #
# ----- Load and prepare image --------- #
# -------------------------------------- #

img = Image.open(input_img)
a = np.array(img)
h, w, c = a.shape

# Flatten
X = a.reshape(h*w, c)

# Get unique colors to upper bound k
n_colors = np.unique(X, axis=0).shape[0]

print(f'Unique colors found in img: {n_colors}')
if k > n_colors:
    print(f'k={k} is greater than the number of unique colors in the image. Setting k to {n_colors}')
    k = n_colors

# Center and scale the data
print(f'Standardizing data...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------- #
# ----- Apply K-Means clustering ------- #
# -------------------------------------- #

print(f'Clustering data...')
kmeans = KMeans(n_clusters=k, random_state=42)  # Adjust 'n_clusters' as needed
kmeans.fit(X_scaled)

# -------------------------------------- #
# ----- Convert back to RGB and save --- #
# -------------------------------------- #

# Replace each pixel by its centroid
X_compressed = kmeans.cluster_centers_[kmeans.labels_]

# Convert back to RGB
print(f'Converting back to RGB and saving...')
img_compressed = scaler.inverse_transform(X_compressed).reshape(h, w, c).astype(np.uint8)
# import matplotlib.pyplot as plt
# plt.imshow(img_compressed)
# plt.show()
img_compressed = Image.fromarray(img_compressed, mode='RGB')
img_compressed.save(output_img)