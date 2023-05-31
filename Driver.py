from keras.datasets import mnist
from OptimalTransportGPU import find_Cost_Between_Images
import matplotlib.pyplot as plt
import numpy as np
from ImageUtility import display_Relation
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# #returns a set of random images associated with the respective answer
# def random_Comparison_Set(set_Images, set_Answer):
#     #set to return of random images of numbers
#     set = ['-']*10

#     satisfy = 0
#     while satisfy < 10:
#         #get rand number
#         rand = random.randint(0, len(set_Answer) - 1)
#         if len(set[set_Answer[rand]]) == 1:
#             set[set_Answer[rand]] = set_Images[rand]
#             satisfy += 1
#     return set

# #method to display a set of images starting at id 0
# #press random key to close all windows
# def display_Set(set, string='number'):
#     id = 0
#     for img in set:
#         str = "{} = {}\n".format(string, id)
#         cv2.imshow(str, img)
#         id += 1
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# #takes a comparison set, a source set, and a key set
# #a random image is chosen from the source set and is then compared to all
# #from the comparison set, the closest is then classified as the orignial image
# def classify_Random_Number(comparison_Set, source_Set, key_Set):
#     total_time = 0
#     rand = random.randint(0, len(source_Set) - 1)
#     rand_Image = source_Set[rand]
#     best_Distance = float('inf')
#     best_Candidite = 0
#     for index in range(0, len(comparison_Set)):
#         comp = comparison_Set[index]
#         result, seconds = compare_Images(comp, rand_Image)
#         print(result)
#         print(key_Set[rand], index, "**\n")
#         if result.cpu().numpy() < best_Distance:
#             best_Distance = result.cpu().numpy()
#             best_Candidite = index
#         total_time += seconds
#     print("\n***\nrandom number {} was classified as {}".format(key_Set[rand], best_Candidite))
#     # cv2.imshow('random', rand_Image)        
#     hours = total_time // 3600
#     total_time -= hours * 3600
#     minutes = total_time // 60
#     total_time -= minutes * 60
#     print("Result took time = {}h {}m {}s\n***\n ".format(hours, minutes, total_time))
#     return (1 if key_Set[rand] == best_Candidite else 0)

# def classify_Random_Number_Convolution(comparison_Set, source_Set, key_Set, set):
#     total_time = 0
#     rand = random.randint(0, len(source_Set) - 1)
#     rand_Image = image_To_Convolutions(source_Set[rand])
#     # cv2.imshow('rand', source_Set[rand])
#     best_Distance = float('inf')
#     best_Candidite = 0
#     for index in range(0, len(comparison_Set)):
#         # cv2.imshow('comp', set[index])
#         comp = comparison_Set[index]
#         result_Sum = 0
#         for i in range(len(comp)):
#             result, seconds = compare_Images(comp[i], rand_Image[i])
#             result_Sum += result.cpu().numpy()
#             total_time += seconds
#             # cv2.imshow('randCOV', rand_Image[i])
#             # cv2.imshow('compCOV', comp[i])
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows();
#         print(result_Sum, key_Set[rand], index, "**\n")
#         if result_Sum < best_Distance:
#             best_Distance = result_Sum
#             best_Candidite = index
#         total_time += seconds
#     print("\n***\nrandom number {} was classified as {}".format(key_Set[rand], best_Candidite))
#     # cv2.imshow('random', rand_Image)        
#     hours = total_time // 3600
#     total_time -= hours * 3600
#     minutes = total_time // 60
#     total_time -= minutes * 60
#     print("Result took time = {}h {}m {}s\n***\n ".format(hours, minutes, total_time))
#     return (1 if key_Set[rand] == best_Candidite else 0)

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


first = x_train[0]
second = x_train[0]
a, b, F, yA, yB, total_cost, iteration, time = find_Cost_Between_Images(first, second, Threshold=50)
fig, axs = plt.subplots(2)
display_Relation(first, second, a, b, F, axs[0])
first = x_train[1]
second = x_train[0]
a, b, F, yA, yB, total_cost, iteration, time = find_Cost_Between_Images(first, second, Threshold=50)
display_Relation(first, second, a, b, F, axs[1])