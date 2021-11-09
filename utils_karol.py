import matplotlib.pyplot as plt
import random
from collections import defaultdict

# name_counter = defaultdict(int)

def save_image(image, name, save_dir="/home/k539i/Documents/tmp/", id=None, note=None):
    global counter
    image = (image - image.min()) / (image.max() - image.min())
    if id is None:
        id = random.randint(0, 10000)
    # counter += 1
    # name_counter[name] += 1
    plt.imshow(image)
    if note is None:
        filename = save_dir + name + "_" + str(id).zfill(4) + ".png"
    else:
        filename = save_dir + name + "_" + str(id).zfill(4) + "_" + note + ".png"
    plt.savefig(filename)
    return id
