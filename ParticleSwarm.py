import pyswarm
from OptimalTransport import POT

def optimal_sample_transform(comp_set, sample_image):
    """
    Takes a comparison set and a sample image
    Performs PSO with POT as a cost descriptor

    Args:
        comp_set (list of images): original forms
        sample_image (image): variation
    """