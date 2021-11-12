import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
from tqdm import tqdm
from medpy.io import load
from natsort import natsorted
import os
import numpy as np
from pathlib import Path
from collections import defaultdict


def convert_dataset(load_dir, save_dir, only_with_label):
    metadata_path = load_dir + "proccessed_data/metadata.csv"
    img_load_dir = load_dir + "proccessed_data/images/"
    img_train_save_dir = save_dir +"train/images/"
    img_val_save_dir = save_dir + "val/images/"
    label_train_save_dir = save_dir + "train/labels/"
    label_val_save_dir = save_dir + "val/labels/"

    Path(img_train_save_dir).mkdir(parents=True, exist_ok=True)
    Path(img_val_save_dir).mkdir(parents=True, exist_ok=True)
    Path(label_train_save_dir).mkdir(parents=True, exist_ok=True)
    Path(label_val_save_dir).mkdir(parents=True, exist_ok=True)

    img_width = 1024
    img_height = 1024
    val_set_ratio = 0.3

    metadata = pd.read_csv(metadata_path)

    positive_set, negative_set = defaultdict(list), defaultdict(list)

    for index, row in metadata.iterrows():
        name, x, y, width, height, label = row["img_name"], row["x"], row["y"], row["width"], row["height"], row["label"]
        if label == 1:
            x /= img_width
            y /= img_height
            width /= img_width
            height /= img_height
            positive_set[name].append({"x": x, "y": y, "width": width, "height": height})
        else:
            negative_set[name].append({"x": x, "y": y, "width": width, "height": height})

    train_set, val_set = train_test_split(list(positive_set.keys()), test_size=val_set_ratio)

    for name in tqdm(train_set):
        copyfile(img_load_dir + name, img_train_save_dir + name)
        with open(label_train_save_dir + name[:-4] + ".txt", 'w') as f:
            for entry in positive_set[name]:
                f.write("{} {} {} {} {} \n".format(0, entry["x"], entry["y"], entry["width"], entry["height"]))

    for name in tqdm(val_set):
        copyfile(img_load_dir + name, img_val_save_dir + name)
        with open(label_val_save_dir + name[:-4] + ".txt", 'w') as f:
            for entry in positive_set[name]:
                f.write("{} {} {} {} {} \n".format(0, entry["x"], entry["y"], entry["width"], entry["height"]))

    if not only_with_label:
        train_set, val_set = train_test_split(list(negative_set.keys()), test_size=val_set_ratio)

        for name in tqdm(train_set):
            copyfile(img_load_dir + name, img_train_save_dir + name)

        for name in tqdm(val_set):
            copyfile(img_load_dir + name, img_val_save_dir + name)


def compute_mean_std(load_dir):
    filenames = os.listdir(load_dir)
    filenames = np.asarray(filenames)
    filenames = natsorted(filenames)

    mean, std, min_value, max_value = 0, 0, 0, 0
    for filename in tqdm(filenames):
        image, _ = load(load_dir + filename)
        mean += image.mean()
        std += image.std()
        if min_value > image.min():
            min_value = image.min()
        if max_value < image.max():
            max_value = image.max()

    mean /= len(filenames)
    std /= len(filenames)

    print("Mean: ", mean)
    print("Std: ", std)
    print("Min: ", min_value)
    print("Max: ", max_value)


if __name__ == "__main__":
    load_dir = "/home/k539i/Documents/datasets/original/node21/"
    save_dir = "/home/k539i/Documents/datasets/preprocessed/node21/"
    convert_dataset(load_dir, save_dir, only_with_label=False)

    # load_dir = "/home/k539i/Documents/datasets/preprocessed/node21/train/images/"
    # compute_mean_std(load_dir)
