import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from collections import defaultdict
import numpy as np
import seaborn as sns


def preprocess(load_file, save_file):
    df = pd.read_csv(load_file)
    df = df.iloc[:, 1:]
    scores = {"Model Scale": [], "mAP": []}
    model_scales = ["N", "S", "M", "L", "X"]
    model_scale_names = ["Nano", "Small", "Medium", "Large", "eXtra large"]
    for name in df:
        for i in range(len(model_scales)):
            if "({})".format(model_scales[i]) in name:
                scores["Model Scale"].append(model_scale_names[i])
                scores["mAP"].append(float(df[name].iloc[0]))
                break

    df = pd.DataFrame(data=scores)

    sns.boxplot(x="Model Scale", y="mAP", data=df)

    plt.ylim(0, 1)

    # plt.show()
    plt.savefig(save_file, dpi=200)


if __name__ == "__main__":
    load_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/preprocessed/"
    save_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/plots/"
    name = "Model Scale.csv"

    preprocess(join(load_dir, name), join(save_dir, name)[:-4] + ".png")
