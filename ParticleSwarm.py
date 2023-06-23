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
        img (array of points): transformed image
    """
    comp_set_extracted = []
    for image in comp_set:
        comp_set_extracted.append(image_Points_Intensities(image))
    
    xopt, fopt = pyswarm.pso(objective_function, lb, ub, 
                             args=(comp_set_extracted, sample_image), 
                             minfunc=1e-5, minstep=1e-4, swarmsize=20, 
                             maxiter=100, debug=False)
    original_cost = objective_function([0, 0, 0, 1, 1, 0, 0], comp_set_extracted, sample_image)
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