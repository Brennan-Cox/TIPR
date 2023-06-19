from contextlib import contextmanager
from io import StringIO
import sys
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import random
from ImageUtility import relation_Figure
from ParticleSwarm import optimal_sample_transform
from OptimalTransport import POT
from tqdm import tqdm

print("MNIST loading...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("loaded")

# returns a set of random images associated with the respective answer
def random_Comparison_Set(set_Images, set_Answer):
    #set to return of random images of numbers
    set = ['-']*10

    satisfy = 0
    while satisfy < 10:
        #get rand number
        rand = random.randint(0, len(set_Answer) - 1)
        if len(set[set_Answer[rand]]) == 1:
            set[set_Answer[rand]] = set_Images[rand]
            # print(rand)
            satisfy += 1
    return set

# returns a random image from the given source set and it's corresponding answer
def random_Image(set_Images, set_Answer):
    rand = random.randint(0, len(set_Images) - 1)
    rand_Image = set_Images[rand]
    rand_Answer = set_Answer[rand]
    
    return rand_Image, rand_Answer

# takes a comp set, image, number of max itterations
# returns the best candidate for the given image, and a list of relations
# between that image and each that it was compared to
# relations have relation per index
# relation has time, cost, plan, a, b
def classify_Image(comparison_Set, Image, reg):
    
    relations = []
    
    best_Distance = float('inf')
    best_Candidate = 0
    for i in range(len(comparison_Set)):
        
        comp_Image = comparison_Set[i]
        
        a, b, cost, total_time, transport_Plan = POT(comp_Image, Image, reg)
        
        # find lowest cost transport plan
        if (cost < best_Distance):
            best_Distance = cost
            best_Candidate = i
            
        relations.append([total_time, cost, transport_Plan, a, b])
    return best_Candidate, relations

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

def testOT(cases, trials_per_case, isPlot, reg):
    data = []
    for i in range(cases):
        totalCorrect = 0
        testCases = trials_per_case
        comparison_Set = random_Comparison_Set(x_test, y_test)
        for i in range(testCases):
            
            rand_Image, rand_Answer = random_Image(x_train, y_train)
            classified_As, relations = classify_Image(comparison_Set, rand_Image, reg)
            
            if (rand_Answer == classified_As):
                totalCorrect = totalCorrect + 1
            
            if (isPlot):
                relation_Figure(comparison_Set, rand_Image, rand_Answer, classified_As, relations)
                
        accuracy = totalCorrect / testCases * 100
        data.append(accuracy)
    sb.displot(data, kde=True, bins=cases)
    
    accuracy = np.sum(data) / cases
    string = "Accuracy of OT is {}%".format(accuracy)
    string += '\nReg {}'.format(reg)
    plt.title(string)
    print(string)

def testPSO(cases, trials_per_case, isPlot, reg):
    data = []
    progressBar = tqdm(total=cases * trials_per_case, desc='testPSO')
    for i in range(cases):
        totalCorrect = 0
        testCases = trials_per_case
        comparison_Set = random_Comparison_Set(x_test, y_test)
        for j in range(testCases):
            
            rand_Image, rand_Answer = random_Image(x_train, y_train)
            with suppress_stdout():
                transformed = optimal_sample_transform(comparison_Set, rand_Image)
            classified_As, relations = classify_Image(comparison_Set, transformed, reg)
            
            if (rand_Answer == classified_As):
                totalCorrect = totalCorrect + 1
            
            if (isPlot):
                relation_Figure(comparison_Set, transformed, rand_Answer, classified_As, relations)
            progressBar.update(1)
        accuracy = totalCorrect / testCases * 100
        data.append(accuracy)
    sb.displot(data, kde=True, bins=cases)
    
    accuracy = np.sum(data) / cases
    string = "Accuracy of OT is {}%".format(accuracy)
    string += '\nReg {}'.format(reg)
    plt.title(string)
    print(string)

testPSO(30, 30, False, 1e-4)