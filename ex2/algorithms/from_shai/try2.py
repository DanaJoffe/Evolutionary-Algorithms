from os.path import join

from sklearn.metrics import recall_score, precision_score
import pandas
import numpy as np
from sklearn.preprocessing import Normalizer as Scaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, Flatten, MaxPool1D, GaussianNoise
from keras.metrics import Precision, Recall, AUC
data_path = "C:\git\Evolutionary-Algorithms\ex2\data"
sample = 100000
#data = np.genfromtxt(, delimiter=',', max_rows=sample)
from algorithms.from_shai.utils import *
data = read_data(join(data_path, "train.csv"), sample=sample)
#data = data.sample(1000)

x, y = data[:,1:], data[:,0]
scaler = Normalizer()

x = scaler.fit_transform(x).reshape(x.shape+(1, ))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

model = Sequential()
model.add(Conv1D(filters=12, kernel_size=4, activation="relu"))
model.add(MaxPool1D(2))
#model.add(GaussianNoise(0.1))
model.add(Conv1D(filters=24, kernel_size=4, activation="relu"))
#model.add(MaxPool1D(2))
#model.add(GaussianNoise(0.1))
model.add(Flatten())
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(100, activation="relu"))
model.add(GaussianNoise(0.15))
model.add(Dropout(0.5))
model.add(Dense(100, activation="relu"))
model.add(Dense(1, activation="relu"))
model.compile(loss="mean_squared_error", optimizer="adam")#, metrics=[Precision(), Recall(), AUC()])
model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test), shuffle=True, verbose=2)


t = model.predict(x_test)

thresholds = [0.01 * i for i in range(1, 100)]
threshold = find_threshold(y_test, t, thresholds, f25)
tp = prediction_by_threshold(t, threshold)

print(f25(y_test, tp), threshold)
exit()











f = []
for i in [0.01*j for j in range(1,100)]:
    classifications = [1 if k > i else 0 for k in t]
    try:
        p = precision_score(y_test, classifications)
        r = recall_score(y_test, classifications)
    except:
        p=0.05
        r=0.05

    f25 = (1 + 0.25**2) * ((p*r)/((0.25**2) * p + r))
    f.append(f25)

print(max(f))





"""
functions = [lambda x:x+i for i in range(100)]

main(i):
    load(models[i])
    model.fit()
"""

# id = asdfasjfhasjkdfhsjdkhfsdjkahasjkdhfasjkdfh






















