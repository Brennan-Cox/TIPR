import matplotlib.pyplot as plt
import openpyxl
import seaborn as sb
import numpy as np
from ImageUtility import image_Points_Intensities, image_To_Outline, relation_Figure, ub, lb
from MNIST import random_Comparison_Set, random_Image
# from MNIST import random_Comparison_Set, random_Image
from OptimalTransport import classify_Image
from ParticleSwarm import objective_function_custom, optimal_sample_transform, optimal_sample_transform_test
from Fonts import get_Random_Set, read_font, transform_image
from tqdm import tqdm
import random
import pandas as pd
from IO import suppress_stdout
from writer import write_To_Excel

def testPSO(cases, trials_per_case, display, display_Incorrect):
    data = []
    rows = []
    cols = ['font', 'answer', 'classified_as']
    options = {
                        'comp_set': None,
                        'sample_image': None,
                        'func': objective_function_custom,
                        'lb': lb,
                        'ub': ub,
                        'swarmsize': 40,
                        'w': 1.0,
                        'c1': 0.5,
                        'c2': 0.5,
                        'maxiter': 100,
                        'minstep': 1e-4,
                        'minfunc': 1e-5,
                        'debug': False,
                        'inertia_decay': 0.96
                    }
    progressBar = tqdm(total=cases * trials_per_case, desc='testPSO')
    for i in range(cases):
        totalCorrect = 0
        testCases = trials_per_case
        # Fonts
        original_Set, font = get_Random_Set(size=30, characters='0123456789')
        # MNIST
        # original_Set = random_Comparison_Set()
        # font = 'MNIST'
        # Convolution
        comparison_Set = original_Set
        for k in range(len(original_Set)):
            comparison_Set[k] = image_To_Outline(original_Set[k])

        for j in range(testCases):
            
            rand_Answer = random.randint(0, len(comparison_Set) - 1)
            rand_Image = transform_image(comparison_Set[rand_Answer])
            # MNIST
            # rand_Image, rand_Answer = random_Image()
            
            # PSO
            try:
                with suppress_stdout():
                    options['comp_set'] = comparison_Set
                    options['sample_image'] = rand_Image
                    tqdm.write('***************BEGIN TEST CASE NUMBER {:}***************'.format(j))
                    transformed_Images, classified_As, xopt = optimal_sample_transform(options)
                    rows.append([font, rand_Answer, classified_As])
            except Exception as e:
                print(e)
                tqdm.write('***************Test case failed with font {}***************'.format(font))
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
        tqdm.write('Accuracy for font {:} is {:}%'.format(font, accuracy))

        path = 'Fonts-swarmSize={:},w={:},c1={:},c2={:},maxIter={:},minStep={:},minFunc={:},inertiaDecay={:}.xlsx'.format(options['swarmsize'], options['w'], options['c1'], options['c2'], options['maxiter'], options['minstep'], options['minfunc'], options['inertia_decay'])
        tqdm.write('Writing to excel file...')
        write_To_Excel(rows=rows, cols=cols, sheetName='results', path=path)
        rows = []
    
    progressBar.close()
    sb.displot(data, kde=True, bins=cases)
    
    accuracy = np.sum(data) / cases
    string = "Accuracy is {}".format(accuracy)
    plt.title(string)

testPSO(100, 100, display=False, display_Incorrect=False)
