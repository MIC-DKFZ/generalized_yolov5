import pandas as pd
from os.path import join


def preprocess(load_file, save_file):
    df = pd.read_csv(load_file)
    df = df.iloc[[499]]
    df = df[df.columns[1::3]]
    df = df.dropna(axis=1)
    df.to_csv(save_file)



if __name__ == "__main__":
    load_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/raw/"
    save_dir = "D:/syncthing-DKFZ/evaluation/yolov5/Node21/preprocessed/"
    name = "Image Size.csv"

    preprocess(join(load_dir, name), join(save_dir, name))
