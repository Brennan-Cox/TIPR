from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from ImageUtility import relation_Figure
import ot, time, random

print("MNIST loading...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("loaded")

def POT(comp_Image, Image, reg):
    
    # get sets of significant points
    a = np.argwhere(comp_Image > 0)
    b = np.argwhere(Image > 0)
    
    SA = comp_Image[a[:, 0], a[:, 1]]
    SA = SA / np.sum(SA)
    
    DB = Image[b[:, 0], b[:, 1]]
    DB = DB / np.sum(DB)
            
    # calculate transport plan and cost using POT
    start_time = time.time()
    cost_Matrix = ot.dist(a, b, 'sqeuclidean')
    cost_Matrix = cost_Matrix / np.max(cost_Matrix)
    transport_Plan = ot.emd(SA, DB, cost_Matrix, reg)
    cost = ot.emd2(SA, DB, cost_Matrix, transport_Plan)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return a, b, cost, total_time, transport_Plan
    

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


def test(cases, trails_per_case, isPlot, reg):
    data = []
    for i in range(cases):
        totalCorrect = 0
        testCases = trails_per_case
        comparison_Set = random_Comparison_Set(x_test, y_test)
        for i in range(testCases):
            
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
    string += '\nReg {}'.format(reg)
    plt.title(string)
    print(string)
    return accuracy

test(1, 1, True, 1e-4)