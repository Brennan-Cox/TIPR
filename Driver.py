from keras.datasets import mnist
from OptimalTransportGPU import find_Cost_Between_Images
import matplotlib.pyplot as plt
import numpy as np
from ImageUtility import display_Relation, display_Set
from PIL import Image
import random
import inflect

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
def classify_Random_Number(comparison_Set, source_Set, key_Set):
    fig, axs = plt.subplots(int((len(comparison_Set) + 2) / 2), 2)
    fig.subplots_adjust(top=3.0, hspace=0)

    axs[0, 0].set_title("Comparison Set")
    display_Set(axs[0, 0], comparison_Set)

    rand = random.randint(0, len(source_Set) - 1)
    rand_Image = source_Set[rand]
    rand_Answer = key_Set[rand]
    axs[0, 1].set_title("Image to identify")
    axs[0, 1].imshow(rand_Image, cmap="gray")

    row = 1
    column = 0
    num_to_word = inflect.engine()

    best_Distance = float('inf')
    best_Candidate = 0
    for i in range(len(comparison_Set)):
        subplot = axs[row, column]
        subplot.set_title(num_to_word.number_to_words(i))
        column += 1
        if (column == 2):
            row += 1
            column = 0

        a, b, F, yA, yB, total_cost, iteration, time, DA, SB = find_Cost_Between_Images(comparison_Set[i], rand_Image, delta=0.5)
        
        cost = total_cost.cpu().numpy()
        if (cost < best_Distance):
            best_Distance = cost
            best_Candidate = i

        subplot.set_xlabel("Calculation took {}s\nWith cost of {}".format(round(time, 4), cost))
        display_Relation(comparison_Set[i], rand_Image, a, b, F, subplot)
    print("Calculation finished, the number {} was classified as {}".format(rand_Answer, best_Candidate))
    # fig.savefig('figures/{}.jpg'.format(num_to_word.number_to_words(random.randint(0, 10000))))
    return best_Candidate == rand_Answer
    
totalCorrect = 0
testCases = 1
for i in range(testCases):
    set = random_Comparison_Set(x_test, y_test)
    result = classify_Random_Number(set, x_train, y_train)
    if (result):
        totalCorrect = totalCorrect + 1
print("Accuracy of OT is {}%".format(totalCorrect / testCases * 100))
    

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