import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from ImageUtility import image_To_Outline, relation_Figure
# from MNIST import random_Comparison_Set, random_Image
from OptimalTransport import classify_Image
from ParticleSwarm import optimal_sample_transform, optimal_sample_transform_test
from Fonts import get_Random_Set, read_font, transform_image
from tqdm import tqdm
import random
from IO import suppress_stdout

def testPSO(cases, trials_per_case, display, display_Incorrect):
    data = []
    progressBar = tqdm(total=cases * trials_per_case, desc='testPSO')
    for i in range(cases):
        totalCorrect = 0
        testCases = trials_per_case
        # Fonts
        # font = 'fonts\\fonts-master\\ofl\\abel\\Abel-Regular.ttf'
        # original_Set = read_font(font, '0123456789', size=30)

        # MNIST
        # comparison_Set = random_Comparison_Set()
        for j in range(testCases):

            original_Set, font = get_Random_Set(size=30)
            # Convolution
            for i in range(len(original_Set)):
                original_Set[i] = image_To_Outline(original_Set[i])
            comparison_Set = original_Set
            
            #Fonts
            rand_Answer = random.randint(0, len(original_Set) - 1)
            rand_Image = transform_image(comparison_Set[rand_Answer])
            
            # MNIST
            # rand_Image, rand_Answer = random_Image()
            
            # PSO
            try:
                with suppress_stdout():
                    print('***************BEGIN TEST CASE NUMBER {}***************'.format(j))
                    transformed_Images, classified_As, xopt = optimal_sample_transform(comparison_Set, rand_Image)
            except Exception as e:
                print(e)
                print('***************Test case failed with font {}***************'.format(font))
                continue

            correct = rand_Answer == classified_As
                        
            if (correct):
                totalCorrect = totalCorrect + 1
            
            if (display or (not correct and display_Incorrect)):
                classified_As, relations = classify_Image(comparison_Set, transformed_Images)
                xoptStr = np.array2string(xopt, precision=2, separator=',', suppress_small=True)    
                title = font + ' ' + xoptStr
                relation_Figure(comparison_Set, rand_Image, rand_Answer, transformed_Images, classified_As, relations, title)
            progressBar.update(1)
        accuracy = totalCorrect / testCases * 100
        data.append(accuracy)
    sb.displot(data, kde=True, bins=cases)
    
    accuracy = np.sum(data) / cases
    string = "Accuracy is {}".format(accuracy)
    plt.title(string)

testPSO(1, 30, display=False, display_Incorrect=True)
