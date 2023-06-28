from PIL import ImageFont, Image, ImageDraw
from ImageUtility import apply_transformations_Reverse, display_Set, lb, ub
import random, os, numpy as np
from ImageUtility import apply_transformations
import matplotlib.pyplot as plt

# @inproceedings{Stutz2019CVPR,
#   title = {Disentangling Adversarial Robustness and Generalization},
#   author = {Stutz, David and Hein, Matthias and Schiele, Bernt},
#   booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
#   publisher = {IEEE Computer Society},
#   year = {2019}
# }

print('Running with FONTS')

# Fonts can be downloaded at https://github.com/davidstutz/disentangling-robustness-generalization
# Fonts without MetaData were excluded
# Note that the directories were deleted for lack of usefulness:
# fonts\fonts-master\vis

def read_font(fn, characters, size=28):
    """
    From: https://davidstutz.de/fonts-a-synthetic-mnist-like-dataset-with-known-manifold/
    Read a font file and generate all letters as images.
 
    :param fn: path to font file as TTF
    :param characters: desired characters to generate
    :param size: desired resolution where resolution is size^2
    :return: images
    :rtype: numpy.ndarray
    """
    
    points = size - size/4
    font = ImageFont.truetype(fn, int(points))
 
    data = []
    for char in characters:
        # new grayscale image all white
        img = Image.new('L', (size, size), 255)
        draw = ImageDraw.Draw(img)
        # get dimensions of the character
        textsize = draw.textbbox((0, 0), text=char, font=font)
        # get center
        text_x = (size - textsize[2] - textsize[0]) // 2
        text_y = (size - textsize[3] - textsize[1]) // 2
        # draw image in center
        draw.text((text_x, text_y), char, font=font)
 
        # convert to np matrix
        matrix = np.array(img)
        # invert
        matrix = 255 - matrix
        # append
        data.append(matrix)
 
    return np.array(data)

def get_random_font_path(directory):
    """
    omitted non metadata.pb directories because no numbers
    Method that will get all .ttf files in the given directory
    and return a random occurrence

    Args:
        directory (string): directory to search

    Returns:
        string: relative path to chosen .ttf file
    """
    
    # Fetch all .ttf files that contain fonts
    ttf_files = []
    for root, dir, files in os.walk(directory):
        if 'METADATA.pb' in files:
            for file in files:
                if file.endswith(".ttf"):
                    ttf_files.append(os.path.join(root, file))
    
    if not ttf_files:
        return None
    
    #choose a random font and return the path to this font
    random_font_path = random.choice(ttf_files)
    return os.path.relpath(random_font_path, directory)

def get_Random_Set(characters = '0123456789'):
    """
    generates the provided characters with a random font

    Args:
        characters (string): desired characters to generate

    Returns:
        images: array of images
    """
    path = get_random_font_path('./fonts')
    images = read_font('./fonts/' + path, characters)
    return images

def transform_image(image):
    """
    applies random transformation to an image

    Args:
        image (image): image to be transformed

    Returns:
        image: randomly transformed
    """
    return apply_transformations(generate_random_set(lb, ub), image)

def transform_image_Reverse(image):
    """
    applies random transformation to an image

    Args:
        image (image): image to be transformed

    Returns:
        image: randomly transformed
    """
    return apply_transformations_Reverse(generate_random_set(lb, ub), image)

def transform_Set(images):
    """
    applies a random transformation per image in the given set

    Args:
        images (images): images to transform

    Returns:
        set of images: transformed
    """
    transformed = []
    for image in images:
        transformed.append(transform_image(image))
    return transformed
        
def generate_random_set(lower_bound, upper_bound):
    """
    generates a random set between the values of two other one dimensional sets

    Args:
        lower_bound (set): lower
        upper_bound (set): upper

    Returns:
        set: random set between lower and upper
    """
    random_set = []
    for lower, upper in zip(lower_bound, upper_bound):
        range = upper - lower
        random_element = lower + random.random() * range
        random_set.append(random_element)
    return random_set

# display_Set(plt, get_Random_Set())