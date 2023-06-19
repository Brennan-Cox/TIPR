import numpy as np
import pyswarm
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from keras.datasets import mnist
import ot
import random

# Load MNIST dataset
(_, _), (X_test, _) = mnist.load_data()
image1 = X_test[random.randint(0, len(X_test) - 1)]
image2 = X_test[random.randint(0, len(X_test) - 1)]

# Compute the significant points and weights of each image
points1 = np.argwhere(image1 > 0)
weights1 = image1[points1[:, 0], points1[:, 1]]
weights1 = weights1 / np.sum(weights1)

points2 = np.argwhere(image2 > 0)
weights2 = image2[points2[:, 0], points2[:, 1]]
weights2 = weights2 / np.sum(weights2)

# Define the objective function to minimize the optimal transport distance
def objective_function(x, points1, weights1, points2, weights2):
    transformed_points = apply_transformations(points1, x)
    M = ot.dist(transformed_points, points2)
    M = M / np.max(M)  # Normalize the distance matrix
    reg = 1e-3
    G = ot.emd(weights1, weights2, M, reg)
    cost = ot.emd2(weights1, weights2, M, G)
    return cost

# Define the transformation function
def apply_transformations(points, x):
    angle = x[0]
    shear = x[1]
    scale = x[2]
    translation = (x[3], x[4])
    
    # Apply affine transformation to points
    transformation_matrix = np.array([[scale*np.cos(angle), -scale*np.sin(angle)+shear],
                                      [scale*np.sin(angle), scale*np.cos(angle)+shear]])
    transformed_points = affine_transform(points, transformation_matrix, offset=translation)
    return transformed_points

# Define the bounds for each parameter
lb = [-np.pi, -1, 0.5, -10, -10]  # Lower bounds for angle, shear, scale, translation (x, y)
ub = [np.pi, 1, 1.5, 10, 10]     # Upper bounds for angle, shear, scale, translation (x, y)

# Use Particle Swarm Optimization to find the optimal transformation parameters
xopt, fopt = pyswarm.pso(objective_function, lb, ub, args=(points1, weights1, points2, weights2))

# Apply the optimal transformations to the points
transformed_points = apply_transformations(points1, xopt)

# Compute the original and new distances
original_distance = objective_function([0, 0, 1, 0, 0], points1, weights1, points2, weights2)
new_distance = fopt

# Plot the original and transformed images
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

axs[0].imshow(image1, cmap='gray')
axs[0].set_title(f'Original Image\nDistance: {original_distance:.4f}')
axs[0].axis('off')

axs[1].imshow(image2, cmap='gray')
axs[1].set_title(f'Target Image\nDistance: {new_distance:.4f}')
axs[1].axis('off')

# Overlay the transformed points on the target image
axs[1].plot(transformed_points[:, 1], transformed_points[:, 0], 'ro', markersize=2)

plt.tight_layout()
plt.show()

    

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Define the image points and pixel intensities
image = x_train[0]
image_points = np.argwhere(image > 0)
pixel_intensities = image[image_points[:, 0], image_points[:, 1]]
points = np.array(image_points)

# Define the rotation angle in degrees
angle = 90  # Example: rotate by 45 degrees

# Convert the angle to radians
theta = np.radians(angle)

# Define the rotation matrix
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

# Apply the rotation to the image points
rotated_points = np.dot(points, rotation_matrix.T)

# Retrieve the rotated image points
rotated_image_points = rotated_points.tolist()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the original image
ax1.scatter(*zip(*image_points), cmap='gray')
ax1.set_title('Original Image')

# Plot the rotated image
ax2.scatter(*zip(*rotated_image_points), cmap='gray')
ax2.set_title('Rotated Image')

# Set the aspect ratio to 'equal' for both subplots
ax1.set_aspect('equal')
ax2.set_aspect('equal')

# Show the plot
plt.show()