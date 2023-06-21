from contextlib import contextmanager
from io import StringIO
import sys
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from ImageUtility import relation_Figure
from MNIST import random_Comparison_Set, random_Image
from OptimalTransport import classify_Image
from ParticleSwarm import optimal_sample_transform
from tqdm import tqdm

@contextmanager
def suppress_stdout():
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
        comparison_Set = random_Comparison_Set()
        for j in range(testCases):
            
            rand_Image, rand_Answer = random_Image()
            # with suppress_stdout():
            transformed = optimal_sample_transform(comparison_Set, rand_Image)
            classified_As, relations = classify_Image(comparison_Set, transformed, reg)
            
            if (rand_Answer == classified_As):
                totalCorrect = totalCorrect + 1
            
            if (isPlot):
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

testPSO(1, 1, True, 1e-4)