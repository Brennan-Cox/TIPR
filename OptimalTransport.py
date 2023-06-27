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
    if (len(a) == 0 or len(b) == 0):
        return a, b, float('infinity'), 0, []
    
    start_time = time.time()
    cost_Matrix = ot.dist(a, b, 'sqeuclidean')
    cost_Matrix = cost_Matrix / np.max(cost_Matrix)
    transport_Plan = ot.emd(SA, DB, cost_Matrix, reg)
    cost = ot.emd2(SA, DB, cost_Matrix, transport_Plan)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return a, b, cost, total_time, transport_Plan

def classify_Image(comparison_Set, transformed_Images, reg):
    """
    takes a comp set, image, regularization parameter
    returns the best candidate for the given image, and a list of relations
    between that image and each that it was compared to
    relations have relation per index (positional)

    Args:
        comparison_Set (array-like set of images): images as comparisons to image
        Image (image): sample image to compare
        reg (number): regularization parameter

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
        
        a, b, cost, total_time, transport_Plan = POT(comp_Image, Image, reg)
        
        # find lowest cost transport plan
        if (cost < best_Distance):
            best_Distance = cost
            best_Candidate = i
            
        relations.append([total_time, cost, transport_Plan, a, b])
    return best_Candidate, relations