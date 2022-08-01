import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from collections import defaultdict
import numpy as np
import seaborn as sns


def preprocess(load_file, save_file, title, scaled=True):
    df = pd.read_csv(load_file)
    df = df.iloc[:, 1:]
    scores = {"Batch Size": [], "mAP": []}
    for name in df:
        for i in range(len(batch_sizes)):
            if "{}".format(batch_sizes[i]) in name:
                scores["Batch Size"].append(batch_size_names[i])
                scores["mAP"].append(float(df[name].iloc[0]))
                break

    df = pd.DataFrame(data=scores)

    palette = {key: color for key, color in zip(keys, key_colors)}
    sns.boxplot(x="Batch Size", y="mAP", data=df, palette=palette)

    if not scaled:
        plt.ylim(0, 1)

    # plt.show()

    if scaled:
        plt.savefig(join(save_file, "scaled", title.replace(" ", "_") + ".png"), dpi=200)
    else:
        plt.savefig(join(save_file, "fixed", title.replace(" ", "_") + ".png"), dpi=200)


if __name__ == "__main__":
    root_dir = "C:/Users/Cookie/Documents/GitKraken/cv_sota/"
    load_dir = "detection/datasets/Node21/experiments/results/"
    save_dir = "detection/datasets/Node21/experiments/box_plots/"
    title = "Batch Size"
    batch_sizes = ["b16", "b32", "b64"]
    batch_size_names = ["16", "32", "64"]
    keys = batch_size_names
    key_colors = ["#b34b4c", "#479a5f", "#338dd8"]

    preprocess(join(root_dir, load_dir, title + ".csv"), join(root_dir, save_dir), title)
