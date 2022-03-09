import pandas as pd
from shutil import copyfile
from tqdm import tqdm
from natsort import natsorted
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil
from PIL import Image
from pydicom import dcmread
import imagesize
import yaml
import SimpleITK as sitk
import argparse
from os.path import join
from sklearn.model_selection import train_test_split


def convert_dataset(config_path):
    config = load_config(config_path)
    fold_paths = prepare_folder_structure(config)
    train_positive_set, train_negative_set, test_positive_set, test_negative_set, labels = preprocess_csv(config)
    preprocess_images_train(config, fold_paths, train_positive_set, train_negative_set)
    if "train_test_split" in config:
        preprocess_images_test(config, fold_paths, test_positive_set, test_negative_set)
    generate_yolo_config(config, labels)


def prepare_folder_structure(config):
    print("Preparing folder structure...")
    shutil.rmtree(config["save_dir"], ignore_errors=True)

    save_dir = config["save_dir"]
    if "train_test_split" in config:
        save_dir = join(config["save_dir"], "train")

    fold_paths = []
    for fold in range(config["cv_folds"]):
        img_fold_save_dir = join(save_dir, "fold_" + str(fold), "images")
        label_fold_save_dir = join(save_dir, "fold_" + str(fold), "labels")
        Path(img_fold_save_dir).mkdir(parents=True, exist_ok=True)
        Path(label_fold_save_dir).mkdir(parents=True, exist_ok=True)
        fold_paths.append({"img_path": img_fold_save_dir, "label_path": label_fold_save_dir})
    return fold_paths


def preprocess_csv(config):
    print("Preprocessing csv...")
    metadata = pd.read_csv(config["metadata_path"])

    positive_set = defaultdict(list)
    negative_set = []
    labels = set()

    for index, row in metadata.iterrows():
        name, x, y, width, height, label = row[config["key_name"]], row[config["key_x"]], row[config["key_y"]], row[config["key_width"]], row[config["key_height"]], row[config["key_label"]]
        labels.add(label)
        if config["add_extension"]:
            name = name + "." + config["add_extension"]
        if label > 0:
            if config["convert_bb_size"]:
                width -= x
                height -= y
            img_width, img_height = read_img_metadata(join(config["img_load_dir"], name))
            x /= img_width
            y /= img_height
            width /= img_width
            height /= img_height
            positive_set[name].append({"label": label - 1, "x": x, "y": y, "width": width, "height": height})
        else:
            negative_set.append(name)

    if "train_test_split" in config:
        train_positive_set, test_positive_set = train_test_split(positive_set, train_size=config["train_test_split"], shuffle=True)
        train_negative_set, test_negative_set = train_test_split(negative_set, train_size=config["train_test_split"], shuffle=True)
        return train_positive_set, train_negative_set, test_positive_set, test_negative_set, labels
    else:
        return positive_set, negative_set, None, None, labels


def preprocess_images_train(config, fold_paths, positive_set, negative_set):
    print("Preprocessing dataset...")

    print("1/2 Processing positive labeled data...")
    keys = split_dataset(list(positive_set.keys()), config["cv_folds"])
    for i, fold in enumerate(tqdm(keys)):
        for name in fold:
            copy_img(join(config["img_load_dir"], name), join(fold_paths[i]["img_path"], name), config["convert2natural_img"])
            with open(join(fold_paths[i]["label_path"], name[:-4] + ".txt"), 'w') as f:
                for entry in positive_set[name]:
                    f.write("{} {} {} {} {} \n".format(entry["label"], entry["x"], entry["y"], entry["width"], entry["height"]))

    print("2/2 Processing negative labeled data...")
    keys = split_dataset(negative_set, config["cv_folds"])
    for i, fold in enumerate(tqdm(keys)):
        for name in fold:
            copy_img(join(config["img_load_dir"], name), join(fold_paths[i]["img_path"], name), config["convert2natural_img"])


def preprocess_images_test(config, save_dir, positive_set, negative_set):
    print("Preprocessing dataset...")

    print("1/2 Processing positive labeled data...")
    for i, fold in enumerate(tqdm(positive_set.keys())):
        for name in fold:
            copy_img(join(config["img_load_dir"], name), join(save_dir, "images", name), config["convert2natural_img"])
            with open(join(save_dir, "labels", name[:-4] + ".txt"), 'w') as f:
                for entry in positive_set[name]:
                    f.write("{} {} {} {} {} \n".format(entry["label"], entry["x"], entry["y"], entry["width"], entry["height"]))

    print("2/2 Processing negative labeled data...")
    keys = split_dataset(negative_set, config["cv_folds"])
    for i, fold in enumerate(tqdm(keys)):
        for name in fold:
            copy_img(join(config["img_load_dir"], name), join(save_dir, "images", name), config["convert2natural_img"])


def generate_yolo_config(config, labels):
    save_dir = config["save_dir"]
    if "train_test_split" in config:
        save_dir = join(config["save_dir"], "train")
    labels.remove(0)
    for fold in range(config["cv_folds"]):
        yolo_config = {}
        yolo_config["path"] = save_dir
        train_folds = list(range(config["cv_folds"]))
        del train_folds[fold]
        train_folds = ["./fold_{}/images".format(train_fold) for train_fold in train_folds]
        yolo_config["train"] = train_folds
        yolo_config["val"] = "./fold_{}/images".format(fold)
        yolo_config["nc"] = len(labels)
        yolo_config["names"] = ["class_{}".format(label) for label in list(labels)]

        with open('{}/fold_{}.yaml'.format(save_dir, fold), 'w') as outfile:
            yaml.dump(yolo_config, outfile, default_flow_style=False)


def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def split_dataset(keys, n_folds):
    keys = np.asarray(keys)
    np.random.shuffle(keys)
    remainder = len(keys) % n_folds
    remainder_keys = keys[-remainder:]
    keys = keys[:-remainder]
    keys = keys.reshape(n_folds, -1)
    keys = [list(fold_keys) for fold_keys in keys]
    keys[-1].extend(list(remainder_keys))
    return keys


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
        image.save(save_filename[:-4] + ".png")
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


def normalize(x, input_range=None, output_range=None):
    if input_range is None:
        input_range = (x.min(), x.max())

    if output_range is None:
        output_range = (0, 1)

    if input_range[0] == input_range[1]:
        return x * 0

    x = (x - input_range[0]) / (input_range[1] - input_range[0])
    x = x * (output_range[1] - output_range[0]) + output_range[0]
    return x


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
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to YAML config.')
    args = parser.parse_args()
    convert_dataset(args.config)

    # load_dir = "/home/k539i/Documents/datasets/original/pneumonia_object_detection/stage_2_train_images/"
    # compute_mean_std(load_dir)
