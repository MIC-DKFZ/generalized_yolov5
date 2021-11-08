import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
from tqdm import tqdm

metadata_path = "/dkfz/cluster/gpu/data/OE0441/k539i/original/node21/proccessed_data/metadata.csv"
img_load_dir = "/dkfz/cluster/gpu/data/OE0441/k539i/original/node21/proccessed_data/images/"
img_train_save_dir = "/dkfz/cluster/gpu/data/OE0441/k539i/preprocessed/node21/train/images/"
img_val_save_dir = "/dkfz/cluster/gpu/data/OE0441/k539i/preprocessed/node21/val/images/"
label_train_save_dir = "/dkfz/cluster/gpu/data/OE0441/k539i/preprocessed/node21/train/labels/"
label_val_save_dir = "/dkfz/cluster/gpu/data/OE0441/k539i/preprocessed/node21/val/labels/"

img_width = 1024
img_height = 1024
val_set_ratio = 0.3

metadata = pd.read_csv(metadata_path)

positive_set, negative_set = [], []

for index, row in metadata.iterrows():
    name, x, y, width, height, label = row["img_name"], row["x"], row["y"], row["width"], row["height"], row["label"]
    if label == 1:
        x /= img_width
        y /= img_height
        width /= img_width
        height /= img_height
        positive_set.append({"name": name, "x": x, "y": y, "width": width, "height": height})
    else:
        negative_set.append({"name": name, "x": x, "y": y, "width": width, "height": height})


train_set, val_set = train_test_split(positive_set, test_size=val_set_ratio)

for entry in tqdm(train_set):
    copyfile(img_load_dir + entry["name"], img_train_save_dir + entry["name"])
    with open(label_train_save_dir + entry["name"][:-4] + ".txt", 'w') as f:
        f.write("{} {} {} {} {}".format(1, entry["x"], entry["y"], entry["width"], entry["height"]))

for entry in tqdm(val_set):
    copyfile(img_load_dir + entry["name"], img_val_save_dir + entry["name"])
    with open(label_val_save_dir + entry["name"][:-4] + ".txt", 'w') as f:
        f.write("{} {} {} {} {}".format(1, entry["x"], entry["y"], entry["width"], entry["height"]))


train_set, val_set = train_test_split(negative_set, test_size=val_set_ratio)

for entry in tqdm(train_set):
    copyfile(img_load_dir + entry["name"], img_train_save_dir + entry["name"])

for entry in tqdm(val_set):
    copyfile(img_load_dir + entry["name"], img_val_save_dir + entry["name"])