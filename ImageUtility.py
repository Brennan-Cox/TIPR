import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import inflect, cv2

def image_Points_Intensities(image):
    """
    Takes an image and returns the points of the image and
    the intensities for each respective point
    
    Please note that images are stored in y, x format

    Args:
        image (array-like image): source image
    """
    a = np.argwhere(image > 30)
        
    SA = image[a[:, 0], a[:, 1]]
    SA = SA / np.sum(SA)
    return a, SA

def image_To_GrayScale(image):
    """
    if image is not already grayscale will convert and return converted

    Args:
        image (image): image to convert

    Returns:
        image (image): image in grayscale
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def image_To_Convolutions(img):
    """
    takes an image and returns a list of resultant images with convolutions applied

    Args:
        img (image): sample image to calculate convolutions

    Returns:
        convolutions (list): list of images representing convolutions
    """
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

def display_Set(subplot, set):
    """
    method will take a set of images and append them together in grayscale

    Args:
        subplot (matplotlib.pyplot): a subplot or plot to display a set within
        set (list): list of images to display
    """
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
    img = cv2.filter2D(src=img, ddepth=-1, kernel=outline)
    return img

def gaussian_filter(kernel_size, img, sigma=1, muu=0):
    """
    creates a gaussian filter of size kernel_size and
    standard deviation sigma then pads with zero to match image shape
    This is necessary to compute the division in frequency domain
    """
    x, y = np.meshgrid(np.linspace(-1,1,kernel_size),
                        np.linspace(-1,1,kernel_size))
    dst = np.sqrt(x**2+y**2)
    normal = 1/(((2*np.pi)**0.5)*sigma)
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    gauss = np.pad(gauss, [(0, img.shape[0] - gauss.shape[0]), (0, img.shape[1] - gauss.shape[1])], 'constant')
    return gauss

def fft_deblur(img, kernel_size, kernel_sigma=5,factor='wiener',const=0.002):
    gauss = gaussian_filter(kernel_size,img,kernel_sigma)
    img_fft = np.fft.fft2(img)
    gauss_fft = np.fft.fft2(gauss)
    weiner_factor = 1 / (1+(const/np.abs(gauss_fft)**2))
    if factor!='wiener':
        weiner_factor = factor
    recon = img_fft/gauss_fft
    recon*=weiner_factor
    recon = np.abs(np.fft.ifft2(recon))
    return recon

def relation_Figure(comparison_Set, image, answer, transformations, classified, relations, title):
    """
    displays the relation figure where an image was proceed

    Args:
        comparison_Set (list): the images used to compare
        rand_Image (image): the sample image compared
        original (image): the image before transformation
        answer (number): the answer on what the image is supposed to be
        classified (number): what the sample image was defined to be
        relations (list of list): list of [total_time, cost, transport_Plan, a, b]
    """
    # size of figure
    rows = int((len(comparison_Set) + 2) / 2) + 1
    fig, axs = plt.subplots(rows, 2)
    fig.subplots_adjust(top=3.0, hspace=0)
    
    axs[rows - 1, 0].set_title(title)
    display_Set(axs[rows - 1, 0], comparison_Set)
    axs[rows - 1, 0].set_title(title)
    axs[rows - 1, 1].remove()

    axs[0, 0].set_title("Comparison Set")
    display_Set(axs[0, 0], comparison_Set)
    
    axs[0, 1].set_title("Image to identify {}".format(answer))
    axs[0, 1].imshow(image, cmap="gray")
    
    # automatic number to word and indexing
    row = 1
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
        subplot.set_xlabel("Calculation took {}s\nWith cost of {}"
                           .format(round(total_time, 4), cost))
        display_Set(subplot, [comparison_Set[i], transformations[i]])
        x_Offset = transformations[i].shape[1]
        for i in range(len(a)):
            for j in range(len(b)):
                if (transport_Plan[i, j] > 0):
                    subplot.plot([a[i, 1], b[j, 1] + x_Offset], 
                                 [a[i, 0], b[j, 0]], linewidth=0.1)
    if (answer == classified):
        fig.set_facecolor("green")
    else:
        fig.set_facecolor("red")
        
#### VECTORS OF TRANSFORMATION ####
scaleLimit = 0.20
rotationLimit = np.degrees(np.pi / 3)
translateLimit = 0.10
shearLimit = 0.20
lb = [-rotationLimit, -translateLimit, -translateLimit, -scaleLimit, -scaleLimit, -shearLimit, -shearLimit]
ub = [rotationLimit, translateLimit, translateLimit, scaleLimit, scaleLimit, shearLimit, shearLimit]

def apply_transformations(x, image):
    """
    Applies transformations based on the dimensions of x

    Args:
        x (list / vector): position of a particle in the swarm
        image (image): original image to be transformed

    Returns:
        image (image): transformed image
    """
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    angle = x[0]
    X_translation = x[1] * width
    Y_translation = x[2] * height
    X_scale = 1 + x[3]
    Y_scale = 1 + x[4]
    X_shear = x[5]
    Y_shear = x[6]
    
    # add rotation axis before scale and shear
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    translation_matrix = np.float32([[1, 0, X_translation], 
                                     [0, 1, Y_translation]])
    image = cv2.warpAffine(image, translation_matrix, (width, height))
    
    scale_matrix = np.float32([[X_scale, 0, 0], 
                               [0, Y_scale, 0]])
    image = cv2.warpAffine(image, scale_matrix, (width, height))
    
    shear_matrix = np.float32([[1, X_shear, 0],
                               [Y_shear, 1, 0]])
    image = cv2.warpAffine(image, shear_matrix, (width, height))
    
    return brighten_image(image)

def brighten_image(image):
    """
    Brightens the image by a random amount

    Args:
        image (image): original image to be brightened

    Returns:
        image (image): brightened image
    """
    # image = image * 2
    # image = np.clip(image, 0, 255)
    return image