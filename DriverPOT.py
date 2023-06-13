from keras.datasets import mnist
import matplotlib.pyplot as plt
from ImageUtility import display_Set, image_To_Outline
import random
import inflect
import seaborn as sb
import numpy as np
import time
import ot

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
        
        a = np.argwhere(comp_Image > 0)
        b = np.argwhere(Image > 0)
        
        start_time = time.time()
        cost_Matrix = ot.dist(a, b, 'sqeuclidean').astype(np.float64)
        cost_Matrix /= np.max(cost_Matrix)
        transport_Plan = ot.emd([], [], cost_Matrix, reg)
        cost = ot.emd2([], [], cost_Matrix, transport_Plan)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if (cost < best_Distance):
            best_Distance = cost
            best_Candidate = i
            
        relations.append([total_time, cost, transport_Plan, a, b])
    return best_Candidate, relations

def relation_Figure(comparison_Set, rand_Image, correct, relations):
    fig, axs = plt.subplots(int((len(comparison_Set) + 2) / 2), 2)
    fig.subplots_adjust(top=3.0, hspace=0)
    axs[0, 0].set_title("Comparison Set")
    display_Set(axs[0, 0], comparison_Set)
    axs[0, 1].set_title("Image to identify")
    axs[0, 1].imshow(rand_Image, cmap="gray")
    #automatic number to word and indexing
    row = 1
    column = 0
    num_to_word = inflect.engine()
    for i in range(len(comparison_Set)):
        subplot = axs[row, column]
        subplot.set_title(num_to_word.number_to_words(i))
        column += 1
        if (column == 2):
            row += 1
            column = 0
        total_time, cost, transport_Plan, a, b = relations[i]
        subplot.set_xlabel("Calculation took {}s\nWith cost of {}".format(round(total_time, 4), cost))
        display_Set(subplot, [comparison_Set[i], rand_Image])
        x_Offset = rand_Image.shape[1]
        for i in range(len(a)):
            for j in range(len(b)):
                if (transport_Plan[i, j] > 0):
                    subplot.plot([a[i, 1], b[j, 1] + x_Offset], [a[i, 0], b[j, 0]], linewidth=0.1)
    if (correct):
        fig.set_facecolor("green")
    else:
        fig.set_facecolor("red")

def test(cases, trails_per_case, isPlot, reg):
    data = []
    for i in range(cases):
        totalCorrect = 0
        testCases = trails_per_case
        for i in range(testCases):
            
            comparison_Set = random_Comparison_Set(x_test, y_test)
            rand_Image, rand_Answer = random_Image(x_train, y_train)
            classified_As, relations = classify_Image(comparison_Set, rand_Image, reg)
            
            if (rand_Answer == classified_As):
                totalCorrect = totalCorrect + 1
            
            if (isPlot):
                relation_Figure(comparison_Set, rand_Image, rand_Answer == classified_As, relations)
                
        accuracy = totalCorrect / testCases * 100
        data.append(accuracy)
    sb.displot(data, kde=True, bins=cases)
    
    accuracy = np.sum(data) / cases
    string = "Accuracy of OT is {}%".format(accuracy)
    plt.title(string + '\nMax Itteration {}'.format(reg))
    print(string)
                
# find_Best_Comp_Set(15)
test(30, 30, False, 1e18)