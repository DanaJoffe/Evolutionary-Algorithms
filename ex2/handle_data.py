import operator
import os
from os.path import join
import pandas as pd
import pathlib


titles = [f'A{j + 1}{i + 1}' for i in range(30) for j in range(4)]
LABEL = 'label'


def load_data(file='train', norm=False):
    data_path = join(os.path.dirname(os.path.abspath(__file__)), f"data/{file}.csv")
    print("=> loading data...")
    data = pd.read_csv(data_path, header=None)
    data.columns = [LABEL] + titles
    print("=> load complete")
    if norm:
        return normalize(data)
    return data


MEAN = 0
STD = 1
train_const = [None, None]


def normalize(dataset):
    print("=> normalize...")
    x = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]
    if train_const[MEAN] is None:
        print("=> normalize... calc mean, calc std...")
        train_const[MEAN] = x.mean(axis=0)
        train_const[STD] = x.std(axis=0)
    dataNorm = (x - train_const[MEAN]) / train_const[STD]
    dataNorm.insert(0, LABEL, y, True)
    print("=> finish normalize.")
    return dataNorm


if __name__ == '__main__':
    df = load_data()
    # y = df.iloc[:, 0]
    # X = df.iloc[:, 1:]
    # for label, (i, row) in zip(y, X.iterrows()):
    for (i, row) in df.iterrows():
        y = row[0]
        x = row[1:].tolist()
        c = 0


    # df = load_data()
    # # df = df.sample(n=2000)
    # y = df.iloc[:, 0]
    # X = df.iloc[:, 1:]
    # for label, (i, row) in zip(y, X.iterrows()):
    #     row = row.tolist()
    #     c= 0




