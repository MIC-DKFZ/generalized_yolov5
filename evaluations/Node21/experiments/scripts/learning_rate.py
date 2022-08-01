import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from collections import defaultdict
import numpy as np
import seaborn as sns


def preprocess(load_file, save_file, title, scaled=True):
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

    palette = {key: color for key, color in zip(keys, key_colors)}
    sns.boxplot(x="Learning Rate", y="mAP", data=df, palette=palette)

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
    title = "Learning Rate"
    keys = [0.001, 0.0005, 0.005, 0.0001, 0.01]
    key_colors = ["#338dd8", "#df672a", "#c1433c", "#3d9e3e", "#edbd3a"]

    preprocess(join(root_dir, load_dir, title + ".csv"), join(root_dir, save_dir), title)
