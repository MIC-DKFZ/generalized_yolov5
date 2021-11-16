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
# from medpy.io import load
from PIL import Image
from pydicom import dcmread
import imagesize
import yaml
import SimpleITK as sitk


def convert_dataset(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    shutil.rmtree(config["save_dir"], ignore_errors=True)
    img_train_save_dir = config["save_dir"] +"train/images/"
    img_val_save_dir = config["save_dir"] + "val/images/"
    label_train_save_dir = config["save_dir"] + "train/labels/"
    label_val_save_dir = config["save_dir"] + "val/labels/"

    Path(img_train_save_dir).mkdir(parents=True, exist_ok=True)
    Path(img_val_save_dir).mkdir(parents=True, exist_ok=True)
    Path(label_train_save_dir).mkdir(parents=True, exist_ok=True)
    Path(label_val_save_dir).mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(config["metadata_path"])

    positive_set = defaultdict(list)
    negative_set = []

    for index, row in metadata.iterrows():
        name, x, y, width, height, label = row[config["key_name"]], row[config["key_x"]], row[config["key_y"]], row[config["key_width"]], row[config["key_height"]], row[config["key_label"]]
        if config["add_extension"]:
            name = name + "." + config["add_extension"]
        if label == 1:
            if config["convert_bb_size"]:
                width -= x
                height -= y
            img_width, img_height = read_img_metadata(config["img_load_dir"] + name)
            x /= img_width
            y /= img_height
            width /= img_width
            height /= img_height
            positive_set[name].append({"x": x, "y": y, "width": width, "height": height})
        else:
            negative_set.append(name)

    train_set, val_set = train_test_split(list(positive_set.keys()), test_size=config["val_set_ratio"])

    for name in tqdm(train_set):
        copy_img(config["img_load_dir"] + name, img_train_save_dir + name, config["convert2natural_img"])
        with open(label_train_save_dir + name[:-4] + ".txt", 'w') as f:
            for entry in positive_set[name]:
                f.write("{} {} {} {} {} \n".format(0, entry["x"], entry["y"], entry["width"], entry["height"]))

    for name in tqdm(val_set):
        copy_img(config["img_load_dir"] + name, img_val_save_dir + name, config["convert2natural_img"])
        with open(label_val_save_dir + name[:-4] + ".txt", 'w') as f:
            for entry in positive_set[name]:
                f.write("{} {} {} {} {} \n".format(0, entry["x"], entry["y"], entry["width"], entry["height"]))

    if not config["only_with_label"]:
        train_set, val_set = train_test_split(negative_set, test_size=config["val_set_ratio"])

        for name in tqdm(train_set):
            copy_img(config["img_load_dir"] + name, img_train_save_dir + name, config["convert2natural_img"])

        for name in tqdm(val_set):
            copy_img(config["img_load_dir"] + name, img_val_save_dir + name, config["convert2natural_img"])


def read_img_metadata(filename):
    extension = filename[-3:]
    if extension in ["dcm", "mha"]:
        try:
            metadata = dcmread(filename, stop_before_pixels=True)
            width = metadata["Columns"].value
            height = metadata["Rows"].value
        except Exception as e:
            image = _load_image(filename)
            height, width = image.shape[:2]
    else:
        width, height = imagesize.get(filename)
    return width, height


def copy_img(load_filename, save_filename, convert2natural_img):
    extension = load_filename[-3:]
    if convert2natural_img and extension in ["dcm", "mha"]:
        image = _load_image(load_filename)
        image = image.squeeze()
        image = (normalize(image) * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(save_filename[:-3] + ".png")
    elif not convert2natural_img:
        copyfile(load_filename, save_filename)
    else:
        raise NotImplementedError("Image format not implemented.")


def _load_image(path):
    im = sitk.GetArrayFromImage(sitk.ReadImage(path))
    # im = im.transpose((2, 1, 0))

    # im, im_header = load(path)
    # im = np.rot90(im, k=-1)
    # im = np.fliplr(im)
    return im


def _save_image(im, filename, compress=False):
    im = sitk.GetImageFromArray((im))
    sitk.WriteImage(im, filename, useCompression=compress)

    # save(im, filename, use_compression=compress)


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
        image = _load_image(load_dir + filename)
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
    config_path = "/home/k539i/Documents/datasets/original/node21/yolov5_convert_config_255.yaml"
    convert_dataset(config_path)

    # load_dir = "/home/k539i/Documents/datasets/original/pneumonia_object_detection/stage_2_train_images/"
    # compute_mean_std(load_dir)
