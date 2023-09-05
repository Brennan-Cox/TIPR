import random
import numpy as np
import pygame
import matplotlib.pyplot as plt
import inflect
import cv2
from IO import suppress_stdout

from ImageUtility import display_Set, image_To_Outline, lb, relation_Figure, ub
from OptimalTransport import classify_Image
from ParticleSwarm import objective_function_custom, optimal_sample_transform

# modified from GFG.com
def prompt_image(i):
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("Draw a " + inflect.engine().number_to_words(i))
    draw_on = False
    last_pos = (0, 0)
    radius = 10
    
    def roundline(srf, color, start, end, radius=1):
        dx = end[0]-start[0]
        dy = end[1]-start[1]
        distance = max(abs(dx), abs(dy))
        for i in range(distance):
            x = int( start[0]+float(i)/distance*dx)
            y = int( start[1]+float(i)/distance*dy)
            pygame.draw.circle(srf, color, (x, y), radius)

    drawing = True
    color = (255, 255, 255)
    while drawing:
        e = pygame.event.wait()
        if e.type == pygame.QUIT:
            drawing = False
        if e.type == pygame.K_RETURN:
            drawing = False
        if e.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.circle(screen, color, e.pos, radius)
            draw_on = True
        if e.type == pygame.MOUSEBUTTONUP:
            draw_on = False
        if e.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(screen, color, e.pos, radius)
                roundline(screen, color, e.pos, last_pos, radius)
            last_pos = e.pos
        pygame.display.flip()
    # convert into a black white image
    image = pygame.surfarray.array3d(screen)
    # swap x and y
    image = image.swapaxes(0, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (30, 30))
    pygame.quit()
    return image

def classify_image(images):
    rand_Answer = random.randint(0, len(images) - 1)
    img = prompt_image(rand_Answer)
    # img = image_To_Outline(img)
    options = {
                        'comp_set': None,
                        'sample_image': None,
                        'func': objective_function_custom,
                        'lb': lb,
                        'ub': ub,
                        'swarmsize': 40,
                        'w': 1.0,
                        'c1': 0.5,
                        'c2': 0.5,
                        'maxiter': 100,
                        'minstep': 1e-4,
                        'minfunc': 1e-5,
                        'debug': False,
                        'inertia_decay': 0.96
                    }
    # PSO
    try:
        with suppress_stdout():
            options['comp_set'] = images
            options['sample_image'] = img
            transformed_Images, classified_As, xopt = optimal_sample_transform(options)
    except Exception as e:
        print(e)

    correct = rand_Answer == classified_As

    classified_As, relations = classify_Image(images, transformed_Images)
    xoptStr = np.array2string(xopt, precision=2, separator=',', suppress_small=True)    
    title = xoptStr
    relation_Figure(images, img, rand_Answer, transformed_Images, classified_As, relations, title)
    return correct
# get all the images
arr = []
for i in range(10):
    img = prompt_image(i)
    # img = image_To_Outline(img)
    arr.append(img)
# display the images
display_Set(plt, arr)

correct = 0
total = 1
for i in range(total):
    if (classify_image(arr)) : correct += 1
print("Accuracy: " + str(correct / total * 100) + "%")