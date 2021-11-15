import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
from tqdm import tqdm
from natsort import natsorted
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil
from medpy.io import load
from PIL import Image
from pydicom import dcmread
import imagesize


def convert_dataset(metadata_path, img_load_dir, save_dir, key_name, key_x, key_y, key_width, key_height, key_label, convert_bb_size, add_extension, only_with_label, convert2natural_img, val_set_ratio):
    shutil.rmtree(save_dir)
    img_train_save_dir = save_dir +"train/images/"
    img_val_save_dir = save_dir + "val/images/"
    label_train_save_dir = save_dir + "train/labels/"
    label_val_save_dir = save_dir + "val/labels/"

    Path(img_train_save_dir).mkdir(parents=True, exist_ok=True)
    Path(img_val_save_dir).mkdir(parents=True, exist_ok=True)
    Path(label_train_save_dir).mkdir(parents=True, exist_ok=True)
    Path(label_val_save_dir).mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)

    positive_set, negative_set = defaultdict(list), defaultdict(list)

    for index, row in metadata.iterrows():
        name, x, y, width, height, label = row[key_name], row[key_x], row[key_y], row[key_width], row[key_height], row[key_label]
        if add_extension is not None:
            name = name + "." + add_extension
        if label == 1:
            if convert_bb_size:
                width -= x
                height -= y
            img_width, img_height = read_img_metadata(img_load_dir + name)
            x /= img_width
            y /= img_height
            width /= img_width
            height /= img_height
            positive_set[name].append({"x": x, "y": y, "width": width, "height": height})
        else:
            negative_set[name].append({"x": x, "y": y, "width": width, "height": height})

    train_set, val_set = train_test_split(list(positive_set.keys()), test_size=val_set_ratio)

    for name in tqdm(train_set):
        copy_img(img_load_dir + name, img_train_save_dir + name, convert2natural_img)
        with open(label_train_save_dir + name[:-4] + ".txt", 'w') as f:
            for entry in positive_set[name]:
                f.write("{} {} {} {} {} \n".format(0, entry["x"], entry["y"], entry["width"], entry["height"]))

    for name in tqdm(val_set):
        copy_img(img_load_dir + name, img_val_save_dir + name, convert2natural_img)
        with open(label_val_save_dir + name[:-4] + ".txt", 'w') as f:
            for entry in positive_set[name]:
                f.write("{} {} {} {} {} \n".format(0, entry["x"], entry["y"], entry["width"], entry["height"]))

    if not only_with_label:
        train_set, val_set = train_test_split(list(negative_set.keys()), test_size=val_set_ratio)

        for name in tqdm(train_set):
            copy_img(img_load_dir + name, img_train_save_dir + name, convert2natural_img)

        for name in tqdm(val_set):
            copy_img(img_load_dir + name, img_val_save_dir + name, convert2natural_img)


def read_img_metadata(filename):
    extension = filename[-3:]
    if extension in ["dcm", "mha"]:
        metadata = dcmread(filename, stop_before_pixels=True)
        width = metadata["Columns"].value
        height = metadata["Rows"].value
    else:
        width, height = imagesize.get(filename)
    return width, height


def copy_img(load_filename, save_filename, convert2natural_img):
    extension = load_filename[-3:]
    if convert2natural_img and extension in ["dcm", "mha"]:
        image, im_header = load(load_filename)
        image = (normalize(image) * 255).astype(np.int)
        image = Image.fromarray(image)
        image.save(save_filename[:-3] + ".png")
    elif not convert2natural_img:
        copyfile(load_filename, save_filename)
    else:
        raise NotImplementedError("Image format not implemented.")


def normalize(x, x_min=None, x_max=None):
    if x_min is None:
        x_min = x.min()

    if x_max is None:
        x_max = x.max()

    if x_min == x_max:
        return x * 0
    else:
        return (x - x.min()) / (x.max() - x.min())


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
    metadata_path = "/home/k539i/Documents/datasets/original/pneumonia_object_detection/stage_2_train_labels.csv"
    img_load_dir = "/home/k539i/Documents/datasets/original/pneumonia_object_detection/stage_2_train_images/"
    save_dir = "/home/k539i/Documents/datasets/preprocessed/pneumonia_object_detection/"
    key_name = "patientId"
    key_x = "x"
    key_y = "y"
    key_width = "width"
    key_height = "height"
    key_label = "Target"
    convert_bb_size = False
    add_extension = "dcm"
    only_with_label = False
    convert2natural_img = False
    val_set_ratio = 0.3

    convert_dataset(metadata_path, img_load_dir, save_dir, key_name, key_x, key_y, key_width, key_height, key_label, convert_bb_size, add_extension, only_with_label, convert2natural_img, val_set_ratio)

    # load_dir = "/home/k539i/Documents/datasets/preprocessed/node21/train/images/"
    # compute_mean_std(load_dir)
