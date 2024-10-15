from PIL import Image
import re
import numpy as np

with open('out.txt', 'r') as f:
    f = f.read()
    print(f)
    results = re.findall(r'Display statement at line 22.+Model has been successfully processed', f, re.DOTALL)[-1]
    results = re.findall(r'(\d+,\d+)', results)

color_lookup = {}
for tup in results:
    point, centroid = tup.split(',')
    color_lookup[int(point)] = int(centroid)

X = np.array(Image.open("./20col.png"))
h, w, c = X.shape

X = X.reshape(-1, 3)

unique_colors = np.unique(X, axis=0) / 255

centroids = []
with open('centroids.txt', 'r') as f:
    for line in f:
        line = line.strip().split(',')
        centroids.append([float(x) for x in line])
centroids = np.array(centroids)
print(centroids)

for i in range(unique_colors.shape[0]):
    print(f'Color {i+1}: {unique_colors[i]}')
    print(f'Assigned to centroid {color_lookup[i+1]}: {centroids[i]}')
    print()

# Replace colors in image according to color lookup
for i in range(unique_colors.shape[0]):
    print(X[np.all(X == unique_colors[i].reshape(1, -1), axis=1)])

X = X.reshape(h, w, c)
import matplotlib.pyplot as plt
plt.imshow(X)
plt.show()