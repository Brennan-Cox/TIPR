import numpy as np
import ot, time
import ot.backend as otb
from ImageUtility import image_Points_Intensities

def L1(a, b):
    """
    Calculates the L1 distance between two images
    the first image is put on top of another image
    then the images are subtracted from each other
    after subtracting the images then the L1 distance is the
    sum of all of the matrix values left

    Args:
        a (np array): set of points describing comp image
        b (np array): set of points describing sample image

    Returns:
        L1 distance between two images
    """
    return np.sum(np.abs(a-b))

def POT(comp_Image, Image):
    """
    Given a comparison image, and sample image, this method will return the EMD OT distance
    between the two images provided

    Args:
        comp_Image (image): image to compare to
        Image (image): image to compare

    Returns:
        POT_Parameterized return values
    """
    # get sets of significant points
    a, SA = image_Points_Intensities(comp_Image)
    b, DB = image_Points_Intensities(Image)
            
    # calculate transport plan and cost using POT
    return POT_Parameterized(a, b, SA, DB)

def POT_Parameterized(a, b, SA, DB):
    """conducts actual POT calculation

    Args:
        a (np array): set of points describing comp image
        b (np array): set of points describing sample image
        SA (np array): normalized weights of comp image pixel intensity
        DB (np array): normalized weights of sample image pixel intensity

    Returns:
        a (np array): set of points describing comp image
        b (np array): set of points describing sample image
        cost (number): the EMD OT distance between the images
        total_time (number): the time OT took to find min distance
        transport_plan (array-like): transport plan for the given cost
    """
    if (len(a) == 0 or len(b) == 0):
        return a, b, float('infinity'), 0, []
    
    start_time = time.time()
    cost_Matrix = ot.dist(x1=a, x2=b, metric='sqeuclidean')
    cost_Matrix = cost_Matrix / np.max(a=cost_Matrix)
    transport_Plan= ot.emd(SA, DB, cost_Matrix)
    # transport_Plan = ot.sinkhorn(a=SA, b=DB, M=cost_Matrix, reg=0.1)
    cost = np.sum(cost_Matrix * transport_Plan)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return a, b, cost, total_time, transport_Plan

def classify_Image(comparison_Set, transformed_Images):
    """
    takes a comp set, image
    returns the best candidate for the given image, and a list of relations
    between that image and each that it was compared to
    relations have relation per index (positional)

    Args:
        comparison_Set (array-like set of images): images as comparisons to image
        Image (image): sample image to compare

    Returns:
        best_Candidate (number): the number that the sample image was classified as
        relations (array-like set of tuples): relation has time, cost, plan, a, b
    """
    
    relations = []
    
    best_Distance = float('inf')
    best_Candidate = 0
    for i in range(len(comparison_Set)):
        
        comp_Image = comparison_Set[i]
        Image = transformed_Images[i]
        
        a, b, cost, total_time, transport_Plan = POT(comp_Image, Image)
        
        # find lowest cost transport plan
        if (cost < best_Distance):
            best_Distance = cost
            best_Candidate = i
            
        relations.append([total_time, cost, transport_Plan, a, b])
    return best_Candidate, relations