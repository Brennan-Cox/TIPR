import numpy as np
import matplotlib.pyplot as plt
import pyswarm
from CustomPSO import custom_pso
from OptimalTransport import POT_Parameterized, L1
from ImageUtility import apply_transformations, image_Points_Intensities, lb, ub
from IO import suppress_stdout
import time

def optimal_sample_transform(options):
    """
    Takes a comparison set and a sample image
    Performs PSO with POT as a cost descriptor
    
    Transformations used:
        Rotation

    Args:
        comp_set (list of images): original forms
        sample_image (image): variation
    Returns:
        best_images (array of images): best transformations obtained
        min_answer (number): the candidate sample was identified as
    """    
    
    best_images = []
    min_score = float('infinity')
    min_answer = 0
    # fit to each pattern
    collisionsArr = []
    itterations = []
    for i in range(len(options['comp_set'])):
        xopt, fopt, collisions, it = custom_pso(func=options['func'], lb=options['lb'], 
                                ub=options['ub'], args=(image_Points_Intensities(options['comp_set'][i]), 
                                options['sample_image']), swarmsize=options['swarmsize'], w=options['w'], 
                                c1=options['c1'], c2=options['c2'],maxiter=options['maxiter'], 
                                minstep=options['minstep'], minfunc=options['minfunc'],
                                debug=options['debug'], inertia_decay=options['inertia_decay'])
        collisionsArr.append(collisions)
        itterations.append(it)
        if (min_score > fopt):
            min_answer = i
            min_score = fopt
        best_images.append(apply_transformations(xopt, options['sample_image']))
    # plt.scatter(itterations, collisionsArr)
    # plt.xlabel("itterations, avg {:}".format(np.average(itterations)))
    # plt.ylabel("collisions, avg {:}".format(np.average(collisionsArr)))
    # m, b = np.polyfit(itterations, collisionsArr, 1)
    # plt.plot(itterations, m*np.array(itterations) + b)
    return best_images, min_answer, xopt

def optimal_sample_transform_test(comp_set, sample_image):
    """
    Takes a comparison set and a sample image
    Performs PSO with POT as a cost descriptor
    
    Transformations used:
        Rotation

    Args:
        comp_set (list of images): original forms
        sample_image (image): variation
    Returns:
        best_images (array of images): best transformations obtained
        min_answer (number): the candidate sample was identified as
    """    
    
    best_images = []
    min_score = float('infinity')
    min_answer = 0
    custom_pso_score = 0
    custom_time = 0
    pyswarm_score = 0
    pyswarm_time = 0
    # fit to each pattern
    for i in range(len(comp_set)):
        # with suppress_stdout():
        start = time.time()
        print("Custom PSO***************")
        xopt, fopt = custom_pso(func=objective_function_custom, lb=lb, ub=ub, 
                                args=(image_Points_Intensities(comp_set[i]), sample_image), 
                                swarmsize=30, w=0.9, c1=0.5, c2=0.5,maxiter=100, 
                                minstep=1e-4, minfunc=1e-5, debug=True, inertia_decay=0.95)
        custom_time += time.time() - start
        start = time.time()
        print("Pyswarm PSO***************")
        with suppress_stdout():
            xopt2, fopt2 = pyswarm.pso(objective_function, lb, ub, 
                                args=(image_Points_Intensities(comp_set[i]), sample_image), 
                                minfunc=1e-6, minstep=1e-4, swarmsize=10, 
                                maxiter=100, debug=True)
        pyswarm_time += time.time() - start
        if (fopt < fopt2):
            custom_pso_score += 1
        else:
            pyswarm_score += 1
            fopt = fopt2
            xopt = xopt2
            
        if (min_score > fopt):
            min_answer = i
            min_score = fopt
        best_images.append(apply_transformations(xopt, sample_image))
    
    print('custom PSO won in {}% of cases'.format(custom_pso_score/len(comp_set)))
    print('custom PSO took {} seconds'.format(custom_time))
    print('pyswarm PSO took {} seconds'.format(pyswarm_time))
    return best_images, min_answer

def objective_function_custom_L1(set_x, comp_extract, image):
    cost_matrix = []
    for x in set_x:
        cost_matrix.append(objective_function_L1(x, comp_extract, image))
    return cost_matrix

def objective_function_L1(x, comp, image):
    image = apply_transformations(x, image)
    return L1(image, comp)

def objective_function_custom(set_x, comp_extract, image):
    """
    This function takes a set of x as a particle swarm vector
    then applies positional transformations in regards to x
    and append the cost of each transformation to a matrix

    Args:
    =====
    set_x (array-like list of numbers): list of numbers that define
        how each transformation in a particle vector is applied
        to the image
    comp_extract ([point, weight]): sample image
    image (image): sample image

    Returns:
    ========
    cost_matrix (array-like list of numbers): list of costs for each
        transformation in the particle vector
    """
    # for each x in set_x call objective_function and return matrix of costs
    cost_matrix = []
    for x in set_x:
        cost_matrix.append(objective_function(x, comp_extract, image))
    return cost_matrix
    
def objective_function(x, comp_extract, image):
    """
    This function takes x as a particle swarm vector
    It then applies positional transformations in regards to x
    
    Args:
    =====
    x (array-like list of numbers): list of numbers that define
        how each transformation in a particle vector is applied
        to the image
    comp_extract ([point, weight]): sample image
    image (image): sample image
    Returns:
    ========
    cost (number): the minimum cost between the transformed image
        and the set of images
    """
    image = apply_transformations(x, image)
    b, DB = image_Points_Intensities(image)
    a, SA = comp_extract
    a, b, cost, total_time, transport_Plan = POT_Parameterized(a, b, SA, DB)
    return cost