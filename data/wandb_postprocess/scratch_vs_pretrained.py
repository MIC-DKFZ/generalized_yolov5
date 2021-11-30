import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from collections import defaultdict
import numpy as np
import seaborn as sns


def preprocess(load_file, save_file):
    df = pd.read_csv(load_file)
    df = df.iloc[:, 1:]
    scratch, pretrained = defaultdict(list), defaultdict(list)
    for name in df:
        sub_names = name.split("_")
        print(name)
        is_scratch = False
        for sub_name in sub_names:
            if "lr0." in sub_name:
                lr = float(sub_name[2:])
            if "Scratch" in sub_name:
                is_scratch = True
        mAP = float(df[name].iloc[0])
        if is_scratch:
            scratch[lr].append(mAP)
        else:
            pretrained[lr].append(mAP)

    def plot(data):
        x = np.asarray(list((data.keys())))
        y = np.asarray([list(scores) for scores in data.values()])
        df = {x[i]:y[i] for i in range(len(x))}
        df = pd.DataFrame.from_dict(df)
        sns.boxplot(data=df)

    plot(scratch)
    plot(pretrained)

    plt.show()



if __name__ == "__main__":
    load_dir = "/home/k539i/Documents/syncthing-DKFZ/evaluation/yolov5/Node21/preprocessed/Scratch vs Pretrained/Scratch vs Pretrained.csv"
    save_dir = "/home/k539i/Documents/syncthing-DKFZ/evaluation/yolov5/Node21/plots/Scratch vs Pretrained/"

    preprocess(load_dir, save_dir)
