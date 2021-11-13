import matplotlib.pyplot as plt
import random
import torch


def save_image(_image, name=None, save_dir="/home/k539i/Documents/tmp/", id=None, note=None, norm=True):
    if isinstance(_image, torch.Tensor):
        image = _image.detach().cpu().numpy()
    else:
        image = _image

    if len(image.shape) == 4:
        image = image[0]

    if image.shape[0] == 1 or image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    if norm:
        image = (image - image.min()) / (image.max() - image.min())

    if id is None:
        id = random.randint(0, 10000)

    plt.imshow(image)

    if name is None:
        filename = save_dir
    else:
        filename = save_dir + name + "_"
    if note is None:
        filename += str(id).zfill(4) + ".png"
    else:
        filename += str(id).zfill(4) + "_" + note + ".png"
    plt.savefig(filename)

    return id
