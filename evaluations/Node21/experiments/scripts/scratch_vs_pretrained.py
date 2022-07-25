import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from collections import defaultdict
import numpy as np
import seaborn as sns


def preprocess(load_file, save_file, title, scaled=True):
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

    def raw2df(data, name):
        x = np.asarray(list((data.keys())))
        y = np.asarray([list(scores) for scores in data.values()])
        x = np.repeat(x, y.shape[1], axis=0)
        y = y.flatten()
        df = {"Learning Rate": x, "mAP": y}
        df = pd.DataFrame(data=df).assign(Initialization=name)
        return df


    scratch = raw2df(scratch, "Scratch")
    pretrained = raw2df(pretrained, "Pretrained")

    df = pd.concat([scratch, pretrained])

    palette = {key: color for key, color in zip(keys, key_colors)}
    sns.boxplot(x="Learning Rate", y="mAP", hue="Initialization", data=df, palette=palette)

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
    title = "Scratch vs Pretrained"
    keys = ["Scratch", "Pretrained"]
    key_colors = ["#df672a", "#338dd8"]

    preprocess(join(root_dir, load_dir, title + ".csv"), join(root_dir, save_dir), title)
