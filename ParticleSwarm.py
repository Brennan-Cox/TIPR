import pyswarm
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
    # fit to each pattern
    for i in range(len(comp_set)):
        xopt, fopt = pyswarm.pso(objective_function, lb, ub, 
                             args=(image_Points_Intensities(comp_set[i]), sample_image), 
                             minfunc=1e-5, minstep=1e-4, swarmsize=10, 
                             maxiter=100, debug=False)
        if (min_score > fopt):
            min_answer = i
            min_score = fopt
        best_images.append(apply_transformations(xopt, sample_image))
    
    return best_images, min_answer
    
def objective_function(x, comp_extract, image):
    """
    This function takes x as a particle swarm vector
    It then applies positional transformations in regards to x
    
    Args:
        x (array-like list of numbers): list of numbers that define
        how each transformation in a particle vector is applied
        comp_extract ([point, weight]): sample image
        image (image): sample image
    Returns:
        cost (number): the minimum cost between the transformed image
        and the set
    """
    # POT reg param
    reg = 1e-4
    image = apply_transformations(x, image)
    b, DB = image_Points_Intensities(image)
    a, SA = comp_extract
    a, b, cost, total_time, transport_Plan = POT_Parameterized(a, b, SA, DB, reg)
    return cost