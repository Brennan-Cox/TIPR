import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import ot
import time

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Choose two random MNIST images
np.random.seed(0)
indices = np.random.randint(0, len(X_train), size=2)
image1 = X_train[indices[0]]
image2 = X_train[indices[1]]

# Compute the significant points and weights of each image
points1 = np.argwhere(image1 > 0)
weights1 = image1[points1[:, 0], points1[:, 1]]
weights1 = weights1 / np.sum(weights1)

points2 = np.argwhere(image2 > 0)
weights2 = image2[points2[:, 0], points2[:, 1]]
weights2 = weights2 / np.sum(weights2)

# Compute the optimal transport between the two images
start_time = time.time()
M = ot.dist(points1, points2)
M = M / np.max(M)  # Normalize the distance matrix
reg = 1e-3
G = ot.emd(weights1, weights2, M, reg)
end_time = time.time()
execution_time = end_time - start_time

# Compute the cost of the optimal transport plan
cost = ot.emd2(weights1, weights2, M, G)

# Visualize the two images overlaid with the optimal transport plan as lines
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

fig.suptitle(f"Optimal Transport Lines\nCalculation Time: {execution_time:.4f} seconds\nCost: {cost:.4f}")

axs[0].imshow(image1, cmap='gray')
axs[0].set_title(f'Image 1\nCost: {cost:.4f}')
axs[0].axis('off')

axs[1].imshow(image2, cmap='gray')
axs[1].set_title(f'Image 2\nCost: {cost:.4f}')
axs[1].axis('off')

axs[2].imshow(image1, cmap='gray')
axs[2].set_title(f'Optimal Transport Lines\nCost: {cost:.4f}')
axs[2].axis('off')

for i in range(len(points1)):
    for j in range(len(points2)):
        if G[i, j] > 0:
            axs[2].plot([points1[i, 1], points2[j, 1]], [points1[i, 0], points2[j, 0]], color='red')

plt.tight_layout()
plt.show()
