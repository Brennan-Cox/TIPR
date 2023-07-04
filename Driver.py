import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from ImageUtility import display_Set, image_To_Outline, relation_Figure
# from MNIST import random_Comparison_Set, random_Image
from OptimalTransport import classify_Image
from ParticleSwarm import optimal_sample_transform
from Fonts import get_Random_Set, transform_Set, transform_image, transform_image_Reverse
from tqdm import tqdm
import random
from IO import suppress_stdout

def testPSO(cases, trials_per_case, display, display_Incorrect, reg):
    data = []
    progressBar = tqdm(total=cases * trials_per_case, desc='testPSO')
    for i in range(cases):
        totalCorrect = 0
        testCases = trials_per_case
        # Fonts
        original_Set, font = get_Random_Set(size=30)
        # Convolution
        for i in range(len(original_Set)):
            original_Set[i] = image_To_Outline(original_Set[i])
        comparison_Set = original_Set
        
        # MNIST
        # comparison_Set = random_Comparison_Set()
        for j in range(testCases):
            
            #Fonts
            rand_Answer = random.randint(0, len(original_Set) - 1)
            rand_Image = transform_image_Reverse(comparison_Set[rand_Answer])
            
            # MNIST
            # rand_Image, rand_Answer = random_Image()
            
            # PSO
            # with suppress_stdout():
            transformed_Images, classified_As = optimal_sample_transform(comparison_Set, rand_Image)
              
            correct = rand_Answer == classified_As
                        
            if (correct):
                totalCorrect = totalCorrect + 1
            
            if (display or (not correct and display_Incorrect)):
                classified_As, relations = classify_Image(comparison_Set, transformed_Images, reg)
                relation_Figure(comparison_Set, rand_Image, rand_Answer, transformed_Images, classified_As, relations)
            progressBar.update(1)
        accuracy = totalCorrect / testCases * 100
        data.append(accuracy)
    sb.displot(data, kde=True, bins=cases)
    
    accuracy = np.sum(data) / cases
    string = "Accuracy is {}\nFont: {}%".format(accuracy, font)
    string += '\nReg {}'.format(reg)
    plt.title(string)

testPSO(1, 1, display=False, display_Incorrect=False, reg=1e-4)