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
    lb = [-45, -translateLimit, -translateLimit]
    ub = [45, translateLimit, translateLimit]
    
    xopt, fopt = pyswarm.pso(objective_function, lb, ub, 
                             args=(comp_set_extracted, sample_image), 
                             minfunc=1e-4, minstep=1e-4, swarmsize=10, 
                             maxiter=100, debug=False)
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
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    translation_matrix = np.float32([[1, 0, X_translation], [0, 1, Y_translation]])
    translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))
    return translated_image