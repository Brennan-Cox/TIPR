import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#if image is not already grayscale will convert and return converted
def image_To_GrayScale(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def image_To_Ndarray(image):
    arr = []
    for x in range(image.shape[0]):
        subarr = []
        for y in range(image.shape[1]):
            subarr.append(image[x][y])
        arr.append(subarr)
    return np.array(arr)

def image_To_Point_Array_SDArray(image, Threshold=100):
    image = image_To_GrayScale(image)
    #list of points
    list = []

    #supply demand list
    sD = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            cell = image[y][x]
            if cell >= Threshold:
                list.append([x, y]) #add point
                sD.append(cell) #add sd
    return list, sD

#returns set a, b, DA, SB
def images_To_Ndarrays(first, second, Threshold=100):

    #incomplete un-normalized and unweighted arrays
    firstList, firstSD = image_To_Point_Array_SDArray(first, Threshold)
    secondList, secondSD = image_To_Point_Array_SDArray(second, Threshold)

    #to np arrays
    firstSD = np.array(firstSD)
    secondSD = np.array(secondSD)

    #weighted versions
    firstSD = firstSD/np.sum(firstSD)
    secondSD = secondSD/np.sum(secondSD)

    return np.array(firstList), np.array(secondList), firstSD, secondSD

#takes an image and returns a list of resultant images with convolutions applied
def image_To_Convolutions(img):
    img = image_To_GrayScale(img)

    horizontal = np.array([[-2, -1, -2],
                       [2, 3, 2],
                       [-2, -1, -2]])
    
    verticle = np.array([[-2, 2, -2],
                     [-1, 3, -1],
                     [-2, 2, -2]])
    
    convolutions = []
    convolutions.append(cv2.filter2D(src=img, ddepth=-1, kernel=horizontal))
    convolutions.append(cv2.filter2D(src=img, ddepth=-1, kernel=verticle))
    return convolutions

def display_Relation(firstImage, secondImage, firstImagePoints, secondImagePoints, transportPlan, subplot):
    height = max(firstImage.shape[0], secondImage.shape[0])
    width = sum([firstImage.shape[1], secondImage.shape[1]])
    newImage = Image.new('L', (width, height))
    newImage.paste(Image.fromarray(firstImage))
    x_Offset = firstImage.shape[1]
    newImage.paste(Image.fromarray(secondImage), (x_Offset, 0))
    subplot.imshow(newImage, cmap='gray')
    # for point in firstImagePoints:
    #     subplot.plot(point[0], point[1], marker='o', color='white')
    # for point in secondImagePoints:
    #     subplot.plot(point[0] + x_Offset, point[1], marker='o', color='white')
    for x in range(firstImagePoints.shape[0]):
        for y in range(secondImagePoints.shape[0]):
            if (transportPlan[y][x] > 0.001):
                point1 = firstImagePoints[x]
                point2 = secondImagePoints[y]
                xVals = [point1[0], point2[0] + x_Offset]
                yVals = [point1[1], point2[1]]
                subplot.plot(xVals, yVals)

#method will take a set of images and append them together
def display_Set(subplot, set):
    height = 0
    width = 0
    for img in set:
        if (img.shape[0] > height):
            height = img.shape[0]
        width += img.shape[1]
    newImage = Image.new('L', (width, height))
    x_Offset = 0
    for img in set:
        newImage.paste(Image.fromarray(img), (x_Offset, 0))
        x_Offset += img.shape[1]
    subplot.imshow(newImage, cmap="gray")