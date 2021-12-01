import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from collections import defaultdict
import numpy as np
import seaborn as sns


def preprocess(load_file, save_file):
    df = pd.read_csv(load_file)
    df = df.iloc[:, 1:]
    scores = {"Batch Size": [], "mAP": []}
    batch_sizes = ["b16", "b32", "b64"]
    batch_size_names = ["16", "32", "64"]
    for name in df:
        for i in range(len(batch_sizes)):
            if "{}".format(batch_sizes[i]) in name:
                scores["Batch Size"].append(batch_size_names[i])
                scores["mAP"].append(float(df[name].iloc[0]))
                break

    df = pd.DataFrame(data=scores)

    sns.boxplot(x="Batch Size", y="mAP", data=df)

    plt.ylim(0, 1)

    # plt.show()
    plt.savefig(save_file, dpi=200)


if __name__ == "__main__":
    load_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/preprocessed/"
    save_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/plots/"
    name = "Batch Size.csv"

    preprocess(join(load_dir, name), join(save_dir, name)[:-4] + ".png")
