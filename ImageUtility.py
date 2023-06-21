import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import inflect

def image_Points_Intensities(image):
    """
    Takes an image and returns the points of the image and
    the intensities for each respective point
    
    Please note that images are stored in y, x format

    Args:
        image (array-like image): source image
    """
    a = np.argwhere(image > 0)
        
    SA = image[a[:, 0], a[:, 1]]
    SA = SA / np.sum(SA)
    return a, SA

#if image is not already grayscale will convert and return converted
def image_To_GrayScale(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

#takes an image and returns a list of resultant images with convolutions applied
def image_To_Convolutions(img):
    img = image_To_GrayScale(img)

    horizontal = np.array([[-2, -1, -2],
                       [2, 3, 2],
                       [-2, -1, -2]])
    
    vertical = np.array([[-2, 2, -2],
                     [-1, 3, -1],
                     [-2, 2, -2]])
    
    convolutions = []
    convolutions.append(cv2.filter2D(src=img, ddepth=-1, kernel=horizontal))
    convolutions.append(cv2.filter2D(src=img, ddepth=-1, kernel=vertical))
    return convolutions

#method will take a set of images and append them together in grayscale
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

def image_To_Outline(img):
    img = image_To_GrayScale(img)

    outline = np.array([[0, -1 , 0],
                        [-1, 4, -1],
                        [0, -1, 0]])

    return cv2.filter2D(src=img, ddepth=-1, kernel=outline)

# displays the relation figure where an image was proceed
def relation_Figure(comparison_Set, rand_Image, original, answer, classified, relations):
    
    #size of figure
    fig, axs = plt.subplots(int((len(comparison_Set) + 2) / 2 + 1), 2)
    fig.subplots_adjust(top=3.0, hspace=0)
    axs[0, 0].set_title("Comparison Set")
    axs[0, 1].remove()
    display_Set(axs[0, 0], comparison_Set)
    axs[1, 0].set_title("Image to identify {}".format(answer))
    axs[1, 0].imshow(original, cmap="gray")
    axs[1, 1].set_title('Transformation')
    axs[1, 1].imshow(rand_Image, cmap='gray')
    #automatic number to word and indexing
    row = 2
    column = 0
    num_to_word = inflect.engine()
    for i in range(len(comparison_Set)):
        subplot = axs[row, column]
        title = ''
        if (i == classified):
            title = 'Classified as: '
        title += num_to_word.number_to_words(i)
        subplot.set_title(title)
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
    if (answer == classified):
        fig.set_facecolor("green")
    else:
        fig.set_facecolor("red")