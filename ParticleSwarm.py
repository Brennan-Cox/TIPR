import pyswarm
from OptimalTransport import POT_Parameterized
from ImageUtility import image_Points_Intensities
import cv2
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

def optimal_sample_transform(comp_set, sample_image):
    """
    Takes a comparison set and a sample image
    Performs PSO with POT as a cost descriptor
    
    Transformations used:
        Rotation

    Args:
        comp_set (list of images): original forms
        sample_image (image): variation
    Returns:
        img (array of points): transformed image
    """
    comp_set_extracted = []
    for image in comp_set:
        comp_set_extracted.append(image_Points_Intensities(image))
        
    translateLimit = 0.25
    lb = [-45, -translateLimit, -translateLimit, 0.8, 0.8]
    ub = [45, translateLimit, translateLimit, 1.2, 1.2]
    
    xopt, fopt = pyswarm.pso(objective_function, lb, ub, 
                             args=(comp_set_extracted, sample_image), 
                             minfunc=1e-4, minstep=1e-4, swarmsize=20, 
                             maxiter=100, debug=True)
    original_cost = objective_function([0, 0, 0, 1, 1], comp_set_extracted, sample_image)
    # best_fopt = float('infinity')
    if (original_cost < fopt):
        return sample_image
    return apply_transformations(xopt, sample_image)
    
def objective_function(x, set, image):
    """
    This function takes x as a particle swarm vector
    It then applies positional transformations in regards to x
    
    Args:
        x (array-like list of numbers): list of numbers that define
        how each transformation in a particle vector is applied
        set (list of [points, weights]): list of sample images
        image (image): sample image
    Returns:
        cost (number): the minimum cost between the transformed image
        and the set
    """
    # POT reg param
    reg = 1e-4
    image = apply_transformations(x, image)
    b, DB = image_Points_Intensities(image)
    minimum_cost = float('infinity')
    time = 0
    for comp_image in set:
        a, SA = comp_image
        a, b, cost, total_time, transport_Plan = POT_Parameterized(a, b, SA, DB, reg)
        minimum_cost = min(cost, minimum_cost)
        time += total_time
    return minimum_cost
    
def apply_transformations(x, image):
    angle = x[0]
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    X_translation = x[1] * width
    Y_translation = x[2] * height
    X_scale = x[3]
    Y_scale = x[4]
    
    #organize orientation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    translation_matrix = np.float32([[1, 0, X_translation], [0, 1, Y_translation]])
    image = cv2.warpAffine(image, translation_matrix, (width, height))
    
    scale_matrix = np.float32([[X_scale, 0, 0], [0, Y_scale, 0]])
    image = cv2.warpAffine(image, scale_matrix, (width, height))
    
    return image

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# image = x_test[0]
# trans = apply_transformations([0, 0, 0, 1.2, 1.2], image)
# fig, axs = plt.subplots(2)
# axs[0].imshow(image, cmap='gray')
# axs[1].imshow(trans, cmap='gray')