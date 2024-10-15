import sys
import numpy as np
from PIL import Image

from kmeans import KMeans
# from sklearn.preprocessing import StandardScaler



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
img = np.array(img)
h, w, c = img.shape

X = img.reshape(h*w, c)

n_colors = np.unique(X, axis=0).shape[0]

print(f'Unique colors found in img: {n_colors}')
if k > n_colors:
    print(f'k={k} is greater than the number of unique colors in the image. Setting k to {n_colors}')
    k = n_colors

# Center and scale the data
# print(f'Standardizing data...')
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = X

# -------------------------------------- #
# ----- Apply K-Means clustering ------- #
# -------------------------------------- #

print(f'Clustering data...')
kmeans = KMeans(n_clusters=k,
                init='random',
                n_init=30,
                normalize=False
                )   # Adjust 'n_clusters' as needed
kmeans.fit(X)

# -------------------------------------- #
# ----- Convert back to RGB and save --- #
# -------------------------------------- #

X_compressed = kmeans.centroids[kmeans.labels]
print(f'Objective value: {kmeans.obj:,.2f}')

# Convert back to RGB
#img_compressed = scaler.inverse_transform(X_compressed).reshape(h, w, c).astype(np.uint8)
# X_compressed = scaler.inverse_transform(X_compressed)
img_compressed = X_compressed.reshape(h, w, c).astype(np.uint8)

print(f'Saving compressed image to {output_img}')
img_compressed = Image.fromarray(img_compressed, mode='RGB')
img_compressed.save(output_img)