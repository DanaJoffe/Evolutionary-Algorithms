import operator
import os
from os.path import join

import pandas as pd
import pathlib


titles = [f'A{j + 1}{i + 1}' for i in range(30) for j in range(4)]
LABEL = 'label'

operators = {operator.add.__name__: '+',
             operator.sub.__name__: '-',
             operator.mul.__name__: '*'}


def load_data():
    data_path = join(os.path.dirname(os.path.abspath(__file__)), "data/train.csv")
    print("=> loading data...")
    data = pd.read_csv(data_path, header=None)
    data.columns = [LABEL] + titles
    print("=> load complete")
    return data


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




