import pyswarm
from CustomPSO import custom_pso
from OptimalTransport import POT_Parameterized
from ImageUtility import apply_transformations, image_Points_Intensities, lb, ub

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
        best_images (array of images): best transformations obtained
        min_answer (number): the candidate sample was identified as
    """    
    
    best_images = []
    min_score = float('infinity')
    min_answer = 0
    custom_pso_score = 0
    pyswarm_score = 0
    # fit to each pattern
    for i in range(len(comp_set)):
        xopt, fopt = custom_pso(func=objective_function_custom, lb=lb, ub=ub, 
                                args=(image_Points_Intensities(comp_set[i]), sample_image), 
                                swarmsize=10, w=0.5, c1=0.5, c2=0.5,maxiter=100, 
                                minstep=1e-4, minfunc=1e-4, debug=False)
        xopt2, fopt2 = pyswarm.pso(objective_function, lb, ub, 
                             args=(image_Points_Intensities(comp_set[i]), sample_image), 
                             minfunc=1e-4, minstep=1e-4, swarmsize=10, 
                             maxiter=100, debug=False)
        
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
    
    # print the percent of times custom_pso was better than pyswarm
    print("Custom PSO was {}% better than PySwarm".format(custom_pso_score / (custom_pso_score + pyswarm_score)))
    return best_images, min_answer

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
    # POT reg param
    reg = 1e-4
    image = apply_transformations(x, image)
    b, DB = image_Points_Intensities(image)
    a, SA = comp_extract
    a, b, cost, total_time, transport_Plan = POT_Parameterized(a, b, SA, DB, reg)
    return cost