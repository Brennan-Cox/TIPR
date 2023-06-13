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

# This method will take a random number from the source set with given actual as key set
# Returns boolean true or false if the number was correctly identified or not
def classify_Random_Number(comparison_Set, source_Set, key_Set, reg, plot=True):
    
    #matplotlib figure with subplots
    if (plot):
        fig, axs = plt.subplots(int((len(comparison_Set) + 2) / 2), 2)
        fig.subplots_adjust(top=3.0, hspace=0)
        axs[0, 0].set_title("Comparison Set")
        display_Set(axs[0, 0], comparison_Set)

    #chose random image and display it
    rand = random.randint(0, len(source_Set) - 1)
    rand_Image = source_Set[rand]
    rand_Image = image_To_Outline(rand_Image)
    rand_Answer = key_Set[rand]
    
    #plot image to id
    if (plot):
        axs[0, 1].set_title("Image to identify")
        axs[0, 1].imshow(rand_Image, cmap="gray")
        #automatic number to word and indexing
        row = 1
        column = 0
        num_to_word = inflect.engine()

    #for each candidate test against random to classify random
    best_Distance = float('inf')
    best_Candidate = 0
    for i in range(len(comparison_Set)):
        if (plot):
            subplot = axs[row, column]
            subplot.set_title(num_to_word.number_to_words(i))
            column += 1
            if (column == 2):
                row += 1
                column = 0

        # print("Transporting...")
        comp_Image = comparison_Set[i]
        comp_Image = image_To_Outline(comp_Image)
        
        a = np.argwhere(comp_Image > 0)
        b = np.argwhere(rand_Image > 0)
        
        start_time = time.time()
        cost_Matrix = ot.dist(a, b, 'sqeuclidean').astype(np.float64)
        cost_Matrix /= np.max(cost_Matrix)
        transport_Plan = ot.emd([], [], cost_Matrix, reg)
        cost = ot.emd2([], [], cost_Matrix, transport_Plan)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # print("Done in {}s".format(total_time))

        if (cost < best_Distance):
            best_Distance = cost
            best_Candidate = i

        if (plot):
            subplot.set_xlabel("Calculation took {}s\nWith cost of {}".format(round(total_time, 4), cost))
            display_Set(subplot, [comp_Image, rand_Image])
            x_Offset = rand_Image.shape[1]
            for i in range(len(a)):
                for j in range(len(b)):
                    if (transport_Plan[i, j] > 0):
                        subplot.plot([a[i, 1], b[j, 1] + x_Offset], [a[i, 0], b[j, 0]], linewidth=0.1)
        
    # print("Calculation finished, the number {} was classified as {}".format(rand_Answer, best_Candidate))
    if (plot):
        # fig.savefig('figures/{}.jpg'.format(num_to_word.number_to_words(random.randint(0, 10000))))
        if (best_Candidate == rand_Answer):
            fig.set_facecolor("green")
        else:
            fig.set_facecolor("red")
    return best_Candidate == rand_Answer
    
def test(cases, trails_per_case, isPlot, reg):
    data = []
    # set = [x_test[6437], x_test[504], x_test[996], x_test[408], x_test[1516], x_test[8192], x_test[1536], x_test[2622], x_test[3737], x_test[3966]]
    # set = [x_test[564], x_test[2693], x_test[3775], x_test[4288], x_test[9550], x_test[710], x_test[8041], x_test[8497], x_test[8843], x_test[4316]]
    for i in range(cases):
        totalCorrect = 0
        testCases = trails_per_case
        for i in range(testCases):
            set = random_Comparison_Set(x_test, y_test)
            result = classify_Random_Number(comparison_Set=set, source_Set=x_train, key_Set=y_train, plot=isPlot, reg=reg)
            if (result):
                totalCorrect = totalCorrect + 1
        accuracy = totalCorrect / testCases * 100
        data.append(accuracy)
    sb.displot(data, kde=True, bins=cases)
    
    accuracy = np.sum(data) / cases
    string = "Accuracy of OT is {}%".format(accuracy)
    plt.title(string + '\nMax Itteration {}'.format(reg))
    print(string)
    
def find_Best_Comp_Set(req_success):
    set = random_Comparison_Set(x_test, y_test)
    best_In_Row = 0
    while (best_In_Row < req_success):
        cand_Set = random_Comparison_Set(x_test, y_test)
        curr_Best_Correct = 0
        cand_Correct = 0
        for i in range(30):
            if classify_Random_Number(comparison_Set=set, source_Set=x_train, key_Set=y_train, plot=False, reg=1e8):
                curr_Best_Correct += 1
            if classify_Random_Number(comparison_Set=cand_Set, source_Set=x_train, key_Set=y_train, plot=False, reg=1e8):
                cand_Correct += 1
        if (cand_Correct > curr_Best_Correct):
            set = cand_Set
            print('Old set got {} new set got {} old set streak {}'.format(curr_Best_Correct, cand_Correct, best_In_Row))
            best_In_Row = 0
        else:
            best_In_Row += 1
    display_Set(plt, set)
    for img in set:
        for i in range(x_test.shape[0]):
            if (np.array_equal(img, x_test[i])):
                print(i)
                
# find_Best_Comp_Set(15)
test(30, 30, False, 1e18)