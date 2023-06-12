from keras.datasets import mnist
from OptimalTransportGPU import find_Cost_Between_Images
import matplotlib.pyplot as plt
from ImageUtility import display_Relation, display_Set
import random
import inflect
import seaborn as sb
import numpy as np

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
def classify_Random_Number(comparison_Set, source_Set, key_Set, plot=True):
    
    #matplotlib figure with subplots
    if (plot):
        fig, axs = plt.subplots(int((len(comparison_Set) + 2) / 2), 2)
        fig.subplots_adjust(top=3.0, hspace=0)
        axs[0, 0].set_title("Comparison Set")
        display_Set(axs[0, 0], comparison_Set)

    #chose random image and display it
    rand = random.randint(0, len(source_Set) - 1)
    rand_Image = source_Set[rand]
    rand_Answer = key_Set[rand]
    
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

        print("Transporting...")
        a, b, F, yA, yB, total_cost, iteration, time, DA, SB = find_Cost_Between_Images(comparison_Set[i], rand_Image, delta=1)
        print("Done in {}s".format(time))

        cost = total_cost.cpu().numpy()
        if (cost < best_Distance):
            best_Distance = cost
            best_Candidate = i

        if (plot):
            subplot.set_xlabel("Calculation took {}s\nWith cost of {}".format(round(time, 4), cost))
            display_Relation(comparison_Set[i], rand_Image, a, b, F, subplot)
        
    print("Calculation finished, the number {} was classified as {}".format(rand_Answer, best_Candidate))
    if (plot):
        # fig.savefig('figures/{}.jpg'.format(num_to_word.number_to_words(random.randint(0, 10000))))
        if (best_Candidate == rand_Answer):
            fig.set_facecolor("green")
        else:
            fig.set_facecolor("red")
    return best_Candidate == rand_Answer
    
def test(cases, trails_per_case):
    data = []
    for i in range(cases):
        totalCorrect = 0
        testCases = trails_per_case
        for i in range(testCases):
            set = random_Comparison_Set(x_test, y_test)
            result = classify_Random_Number(comparison_Set=set, source_Set=x_train, key_Set=y_train, plot=False)
            if (result):
                totalCorrect = totalCorrect + 1
        accuracy = totalCorrect / testCases * 100
        data.append(accuracy)
    sb.displot(data, kde=True, bins=cases)
    
    accuracy = np.sum(data) / cases
    string = "Accuracy of OT is {}%".format(accuracy)
    print(string)
    lines = [string]
    with open('output.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write("\n")
    
test(10, 100)

# def convolution_Of_Set(set):
#     result = []
#     for image in set:
#         convs = image_To_Convolutions(image)
#         result.append(convs)
#     return result    

# def test(itterations, convolution=False):
#     set = random_Comparison_Set(x_train, y_train)
#     convolutionSet = convolution_Of_Set(set)
#     success = 0
#     for i in range(itterations):
#         if (convolution):
#             success += classify_Random_Number_Convolution(convolutionSet, x_train, y_train, set)
#         else:
#             success += classify_Random_Number(set, x_train, y_train)
#         # display_Set(set)
#     return success / itterations