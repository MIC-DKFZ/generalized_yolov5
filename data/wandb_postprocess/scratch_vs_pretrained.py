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

    sns.boxplot(x="Learning Rate", y="mAP", hue="Initialization", data=df)

    plt.ylim(0, 1)

    # plt.show()
    plt.savefig(save_file, dpi=200)


if __name__ == "__main__":
    load_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/preprocessed/"
    save_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/plots/"
    name = "Scratch vs Pretrained.csv"

    preprocess(join(load_dir, name), join(save_dir, name)[:-4] + ".png")
