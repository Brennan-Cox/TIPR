import numpy as np
import ot, time

def POT(comp_Image, Image, reg):
    """
    Given a comparison image, and sample image, as well as
    regularization param, this method will return the EMD OT distance
    between the two images provided

    Args:
        comp_Image (image): image to compare to
        Image (image): image to compare
        reg (number): regularization param, passed into ot

    Returns:
        a (np array): set of points describing comp image
        b (np array): set of points describing sample image
        cost (number): the EMD OT distance between the images
        total_time (number): the time OT took to find min distance
        transport_plan (array-like): transport plan for the given cost
    """
    # get sets of significant points
    a = np.argwhere(comp_Image > 0)
    b = np.argwhere(Image > 0)
    
    SA = comp_Image[a[:, 0], a[:, 1]]
    SA = SA / np.sum(SA)
    
    DB = Image[b[:, 0], b[:, 1]]
    DB = DB / np.sum(DB)
            
    # calculate transport plan and cost using POT
    start_time = time.time()
    cost_Matrix = ot.dist(a, b, 'sqeuclidean')
    cost_Matrix = cost_Matrix / np.max(cost_Matrix)
    transport_Plan = ot.emd(SA, DB, cost_Matrix, reg)
    cost = ot.emd2(SA, DB, cost_Matrix, transport_Plan)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return a, b, cost, total_time, transport_Plan