import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from collections import defaultdict
import numpy as np
import seaborn as sns


def preprocess(load_file, save_file):
    df = pd.read_csv(load_file)
    df = df.iloc[:, 1:]
    scores = {"Standardized vs Not-Standardized": [], "mAP": []}
    for name in df:
        if "standardized" in name:
            scores["Standardized vs Not-Standardized"].append("Standardized")
            scores["mAP"].append(float(df[name].iloc[0]))
        else:
            scores["Standardized vs Not-Standardized"].append("Not-Standardized")
            scores["mAP"].append(float(df[name].iloc[0]))

    df = pd.DataFrame(data=scores)

    sns.boxplot(x="Standardized vs Not-Standardized", y="mAP", data=df)

    plt.ylim(0, 1)

    # plt.show()
    plt.savefig(save_file, dpi=200)


if __name__ == "__main__":
    load_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/preprocessed/"
    save_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/plots/"
    name = "Standardized.csv"

    preprocess(join(load_dir, name), join(save_dir, name)[:-4] + ".png")
