from PIL import ImageFont, Image, ImageDraw
import random, os, numpy as np
from ImageUtility import apply_transformations, lb, ub
import matplotlib.pyplot as plt
from fontTools.ttLib import TTFont

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

def write_valid_font_paths(characters, directory):
    """
    Write the paths to all .ttf files in the provided directory that contain the provided characters to a file.

    Args:
        characters (string): characters to check for
        directory (string): directory to start search in
    """
    print('finding valid fonts')
    # get all .ttf files in the directory that contain the desired characters
    ttf_fonts = []
    # walk through the directory
    for root, dir, files in os.walk(directory):
        # for each file in the directory
        for file in files:
            # if the file is a .ttf file
            if file.endswith(".ttf"):
                
                # get the path to the file
                path = os.path.join(root, file)
                # load the font
                font = TTFont(path)
                # check if the font contains all characters
                for character in characters:
                    for table in font['cmap'].tables:
                        if ord(character) in table.cmap.keys():
                            ttf_fonts.append(path)
                            break
    print('writing valid fonts')
    # write the character set to a file
    with open('characters.txt', 'w') as f:
        f.write(characters)
    # write the valid font paths to a file
    with open('valid_fonts.txt', 'w') as f:
          f.write('\n'.join(ttf_fonts))
    print('done')    
                    
def get_random_font_path(directory, characters):
    """
    Fetch a random font from the provided directory that contains the provided characters.
    
    Args:
        directory (string): directory to search

    Returns:
        string: relative path to chosen .ttf file
    """
    # if the file 'valid_fonts.txt' does not exist, write all valid font paths to this file
    # also write the fonts if the character set has changed
    file_found = os.path.isfile('valid_fonts.txt')
    
    if (os.path.isfile('characters.txt')):
        characters_changed = open('characters.txt', 'r').read() != characters
    else:
        characters_changed = True
            
    if not file_found or characters_changed:
        write_valid_font_paths(characters, directory)
    
    ttf_fonts = []
    # read all valid font paths from the file 'valid_fonts.txt'
    with open('valid_fonts.txt', 'r') as f:
        ttf_fonts = f.read().splitlines()
                    
    if not ttf_fonts:
        return None

    #choose a random font and return the path to this font
    random_font = random.choice(ttf_fonts)
    return os.path.relpath(random_font, directory)

def get_Random_Set(characters = '0123456789', size = 28):
    """
    generates the provided characters with a random font

    Args:
        characters (string): desired characters to generate

    Returns:
        images: array of images
    """
    # There is an os issue with some invalid font, must correct
    # For now this is a valid work around
    Error = True
    while(Error):
        path = get_random_font_path('./fonts', characters)
        try:
            images = read_font('./fonts/' + path, characters, size)
            Error = False
        except OSError:
            print('Invalid font stack overflow for characters {}'.format(path))
            Error = True
            
    return images, path

def transform_image(image):
    """

    Args:
        image (image): image to be transformed

    Returns:
        image: randomly transformed
    """
    return apply_transformations(generate_random_set(lb, ub), image)

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