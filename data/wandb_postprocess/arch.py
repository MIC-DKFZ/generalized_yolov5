import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from collections import defaultdict
import numpy as np
import seaborn as sns


def preprocess(load_file, save_file):
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

    sns.boxplot(x="Arch-P5 vs Arch-P6", y="mAP", data=df)

    plt.ylim(0, 1)

    # plt.show()
    plt.savefig(save_file, dpi=200)


if __name__ == "__main__":
    load_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/preprocessed/"
    save_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/plots/"
    filename = "Arch-P5 vs Arch-P6.csv"

    preprocess(join(load_dir, filename), join(save_dir, filename)[:-4] + ".png")
