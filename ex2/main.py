from algorithms.dt import dt
from algorithms.gp_from_scratch import run_gp_algo
import pandas as pd


# def load_data():
#     df = pd.read_csv("data/train.csv")
#     x = 0


def main():
    # LABEL = 'label'
    # df = pd.read_csv("data/train.csv")
    # # y = df.iloc[:, 0]
    #
    # # take only features
    # # df = df.iloc[:, 1:]
    #
    # titles = [f'{j + 1}' + f'{i + 1}' for i in range(30) for j in range(4)]
    # df.columns = [LABEL] + titles
    # # rows = df.shape[0]
    #
    # df = df.sample(n=100).to_dict('records')
    #
    #
    # dataset = [{'x': x,
    #             'y': y,
    #             LABEL: z}
    #            for x, y, z in [(3, 6, 16), (4, 12, 45), (5, 10, 48), (2, 9, 13.5)]
    #            ]
    #


    # load_data()
    """ choose algorithm from algorithms files & run it"""
    run_gp_algo()


if __name__ == '__main__':
    # variables = [f'{j + 1}' + f'{i + 1}' for i in range(30) for j in range(4)]
    # print(variables)
    # main()
    dt()



