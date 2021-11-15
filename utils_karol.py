import matplotlib.pyplot as plt
import random
import torch
import cv2
import numpy as np


def save_image(_image, _targets=None, name=None, save_dir="/home/k539i/Documents/tmp/", norm=True, only_save_with_label=True):
    if isinstance(_image, torch.Tensor):
        image = _image.detach().cpu().numpy()
    else:
        image = _image

    if _targets is not None:
        if isinstance(_targets, torch.Tensor):
            targets = _targets.detach().cpu().numpy()
        else:
            targets = _targets

    if len(image.shape) == 4:
        image = image[0]

    if image.shape[0] == 1 or image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    if norm:
        image = (normalize(image) * 255).astype(np.uint8).copy()

    target = []
    if targets is not None:
        height, width = image.shape[:2]
        for i in range(len(targets)):
            if targets[i][0] == 0:
                target.append({"x": int(targets[i][2] * width), "y": int(targets[i][3] * height), "width": int(targets[i][4] * width), "height": int(targets[i][5] * height)})

    if only_save_with_label and not target:
        return -1

    if targets is not None:
        for bb in target:
            cv2.rectangle(image, (bb["x"], bb["y"]), (bb["x"] + bb["width"], bb["y"] + bb["height"]), color=(255, 0, 0), thickness=2)

    plt.imshow(image)
    plt.title(name)

    if name is not None:
        filename = save_dir + name[:-4] + ".png"
    else:
        id = random.randint(1, 10000)
        filename = save_dir + str(id).zfill(4) + ".png"

    plt.savefig(filename)


def normalize(x, x_min=None, x_max=None):
    if x_min is None:
        x_min = x.min()

    if x_max is None:
        x_max = x.max()

    if x_min == x_max:
        return x * 0
    else:
        return (x - x.min()) / (x.max() - x.min())