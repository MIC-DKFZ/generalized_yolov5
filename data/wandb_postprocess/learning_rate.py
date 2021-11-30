import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from collections import defaultdict
import numpy as np
import seaborn as sns


def preprocess(load_file, save_file):
    df = pd.read_csv(load_file)
    df = df.iloc[:, 1:]
    learning_rate = defaultdict(list)
    for name in df:
        sub_names = name.split("_")
        print(name)
        for sub_name in sub_names:
            if "lr0." in sub_name:
                lr = float(sub_name[2:])
        mAP = float(df[name].iloc[0])
        learning_rate[lr].append(mAP)

    def raw2df(data):
        x = np.asarray(list((data.keys())))
        y = np.asarray([list(scores) for scores in data.values()])
        x = np.repeat(x, y.shape[1], axis=0)
        y = y.flatten()
        df = {"Learning Rate": x, "mAP": y}
        df = pd.DataFrame(data=df)
        return df


    df = raw2df(learning_rate)

    sns.boxplot(x="Learning Rate", y="mAP", data=df)

    plt.ylim(0, 1)

    # plt.show()
    plt.savefig(save_file, dpi=200)


if __name__ == "__main__":
    load_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/preprocessed/"
    save_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/plots/"
    name = "Learning Rate.csv"

    preprocess(join(load_dir, name), join(save_dir, name)[:-4] + ".png")
