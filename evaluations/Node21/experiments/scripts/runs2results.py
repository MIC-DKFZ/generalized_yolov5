import pandas as pd
from os.path import join


def preprocess(load_file, save_file):
    df = pd.read_csv(load_file)
    df = df.iloc[[499]]
    df = df[df.columns[1::3]]
    df = df.dropna(axis=1)
    df.to_csv(save_file)



if __name__ == "__main__":
    root_dir = "C:/Users/Cookie/Documents/GitKraken/cv_sota/"
    load_dir = "detection/datasets/Node21/experiments/results_raw/"
    save_dir = "detection/datasets/Node21/experiments/results/"
    name = "Image Size.csv"

    preprocess(join(root_dir, load_dir, name), join(root_dir, save_dir, name))
