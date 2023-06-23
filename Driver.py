from contextlib import contextmanager
from io import StringIO
import sys
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from ImageUtility import display_Set, image_To_Outline, relation_Figure
# from MNIST import random_Comparison_Set, random_Image
from OptimalTransport import classify_Image
from ParticleSwarm import optimal_sample_transform
from Fonts import get_Random_Set, transform_Set, transform_image
from tqdm import tqdm
import random
import time

@contextmanager
def suppress_stdout():
    """
    Method when combined with (with:)
    will not let the code within it's section
    output to standard out
    """
    # Create a StringIO object to capture the output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        # Restore the original standard output
        sys.stdout = old_stdout

def testPSO(cases, trials_per_case, isPlot, reg):
    data = []
    progressBar = tqdm(total=cases * trials_per_case, desc='testPSO')
    for i in range(cases):
        totalCorrect = 0
        testCases = trials_per_case
        # Fonts
        original_Set = get_Random_Set()
        fig, axs = plt.subplots(2)
        for i in range(len(original_Set)):
            original_Set[i] = image_To_Outline(original_Set[i])
        comparison_Set = transform_Set(original_Set)
        
        # MNIST
        # comparison_Set = random_Comparison_Set()
        for j in range(testCases):
            
            #Fonts
            rand_Answer = random.randint(0, len(original_Set) - 1)
            rand_Image = transform_image(original_Set[rand_Answer])
            
            # MNIST
            # rand_Image, rand_Answer = random_Image()
            
            # PSO
            # start = time.time()
            with suppress_stdout():
                transformed, classified_As = optimal_sample_transform(comparison_Set, rand_Image)
            # print('PSO took {}s'.format(time.time() - start))
                        
            if (rand_Answer == classified_As):
                totalCorrect = totalCorrect + 1
            
            if (isPlot):
                classified_As, relations = classify_Image(comparison_Set, transformed, reg)
                relation_Figure(comparison_Set, transformed, rand_Image, rand_Answer, classified_As, relations)
                # classified_As, relations = classify_Image(comparison_Set, rand_Image, reg)
                # relation_Figure(comparison_Set, rand_Image, rand_Image, rand_Answer, classified_As, relations)
            progressBar.update(1)
        accuracy = totalCorrect / testCases * 100
        data.append(accuracy)
    sb.displot(data, kde=True, bins=cases)
    
    accuracy = np.sum(data) / cases
    string = "Accuracy of OT is {}%".format(accuracy)
    string += '\nReg {}'.format(reg)
    plt.title(string)

testPSO(1, 30, False, 1e-4)