import numpy as np
import ot, time
from ImageUtility import image_Points_Intensities

def POT(comp_Image, Image, reg):
    """
    Given a comparison image, and sample image, as well as
    regularization param, this method will return the EMD OT distance
    between the two images provided

    Args:
        comp_Image (image): image to compare to
        Image (image): image to compare
        reg (number): consult ot.emd

    Returns:
        POT_Parameterized return values
    """
    # get sets of significant points
    a, SA = image_Points_Intensities(comp_Image)
    b, DB = image_Points_Intensities(Image)
            
    # calculate transport plan and cost using POT
    return POT_Parameterized(a, b, SA, DB, reg)

def POT_Parameterized(a, b, SA, DB, reg):
    """conducts actual POT calculation

    Args:
        a (np array): set of points describing comp image
        b (np array): set of points describing sample image
        SA (np array): normalized weights of comp image pixel intensity
        DB (np array): normalized weights of sample image pixel intensity
        reg (number): consult ot.emd

    Returns:
        a (np array): set of points describing comp image
        b (np array): set of points describing sample image
        cost (number): the EMD OT distance between the images
        total_time (number): the time OT took to find min distance
        transport_plan (array-like): transport plan for the given cost
    """
    start_time = time.time()
    cost_Matrix = ot.dist(a, b, 'sqeuclidean')
    cost_Matrix = cost_Matrix / np.max(cost_Matrix)
    transport_Plan = ot.emd(SA, DB, cost_Matrix, reg)
    cost = ot.emd2(SA, DB, cost_Matrix, transport_Plan)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return a, b, cost, total_time, transport_Plan