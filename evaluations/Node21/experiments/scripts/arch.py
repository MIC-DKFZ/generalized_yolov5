import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from collections import defaultdict
import numpy as np
import seaborn as sns


def preprocess(load_file, save_file, title, scaled=True):
    df = pd.read_csv(load_file)
    df = df.iloc[:, 1:]
    scores = {"Arch-P5 vs Arch-P6": [], "mAP": []}
    ids = ["Arch-P5", "Arch-P6"]
    id_names = ["Arch-P5", "Arch-P6"]
    for name in df:
        for i in range(len(ids)):
            if "{}".format(ids[i]) in name  :
                scores["Arch-P5 vs Arch-P6"].append(id_names[i])
                scores["mAP"].append(float(df[name].iloc[0]))
                break

    df = pd.DataFrame(data=scores)

    palette = {key: color for key, color in zip(keys, key_colors)}
    sns.boxplot(x="Arch-P5 vs Arch-P6", y="mAP", data=df, palette=palette)

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
    title = "Arch-P5 vs Arch-P6"
    keys = ["Arch-P5", "Arch-P6"]
    key_colors = ["#338dd8", "#df672a"]

    preprocess(join(root_dir, load_dir, title + ".csv"), join(root_dir, save_dir), title)
