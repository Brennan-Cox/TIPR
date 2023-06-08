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

#returns a set of random images associated with the respective answer
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

def classify_Random_Number(comparison_Set, source_Set, key_Set):
    fig, axs = plt.subplots(int((len(comparison_Set) + 2) / 2), 2)
    fig.subplots_adjust(top=4.0, hspace=0)

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

        a, b, F, yA, yB, total_cost, iteration, time, DA, SB = find_Cost_Between_Images(comparison_Set[i], rand_Image, delta=0.9)
        
        cost = total_cost.cpu().numpy()
        if (cost < best_Distance):
            best_Distance = cost
            best_Candidate = i

        subplot.set_xlabel("Calculation took {}s\nWith cost of {}".format(round(time, 4), cost))
        display_Relation(comparison_Set[i], rand_Image, a, b, F, subplot)
    print("Calculation finished, the number {} was classified as {}".format(rand_Answer, best_Candidate))
    return best_Candidate == rand_Answer
    
classify_Random_Number(random_Comparison_Set(x_test, y_test), x_train, y_train)

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

def test(first, second, subplot):
    print("transporting...")
    a, b, F, yA, yB, total_cost, iteration, time, DA, SB = find_Cost_Between_Images(first, second, Threshold=50)
    print("Calculation time took: {}s".format(time))
    display_Relation(first, second, a, b, F, subplot)
    # array = np.array(F)
    # print("result")
    # print("how much is each supply node supplying?")
    # for i in range(len(a)):
    #     print(np.sum(array[i]), SB[i])
    # print("how much is each demand node receiving?")
    # for i in range(len(b)):
    #     sum = 0
    #     for k in range(len(a)):
    #         sum += array[k][i]
    #     print(sum, DA[i])
    # print(np.sum(array))
    print("Done")

# fig, axs = plt.subplots(2)
# # first = x_train[0]
# # second = x_train[0]
# # test(first, second, axs[0])
# first = x_train[3]
# second = x_train[1]

# height = max(first.shape[0], second.shape[0])
# width = sum([first.shape[1], second.shape[1]])
# newImage = Image.new('L', (width, height))
# newImage.paste(Image.fromarray(first))
# x_Offset = first.shape[0]
# newImage.paste(Image.fromarray(second), (x_Offset, 0))
# axs[1].imshow(newImage, cmap='gray')
# # test(first, second, axs[1])
# firstC = image_To_Convolutions(first)
# secondC = image_To_Convolutions(second)
# test(first, second, axs[0])
# # test(firstC[0], secondC[0], axs[1])
# # test(firstC[1], secondC[1], axs[2])

