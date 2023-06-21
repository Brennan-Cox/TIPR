from keras.datasets import mnist
import random

print("MNIST loading...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("loaded")

def random_Comparison_Set():
    """
    Built for MNIST processes, this function returns 10 random numbers 0-9

    Args:
        set_Images (array-like set of images): set of images to chose from
        set_Answer (_type_): set of answers for the given images positionally correspondent

    Returns:
        array-like set of images: set of images 0-9 positionally correspondent
    """
    #set to return of random images of numbers
    set = ['-']*10

    satisfy = 0
    while satisfy < 10:
        #get rand number
        rand = random.randint(0, len(y_test) - 1)
        if len(set[y_test[rand]]) == 1:
            set[y_test[rand]] = x_test[rand]
            # print(rand)
            satisfy += 1
    return set

# returns a random image from the given source set and it's corresponding answer
def random_Image():
    rand = random.randint(0, len(x_train) - 1)
    rand_Image = x_train[rand]
    rand_Answer = y_train[rand]
    
    return rand_Image, rand_Answer