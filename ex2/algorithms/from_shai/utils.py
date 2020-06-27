def prediction_by_threshold(t, threshold):
    tp = [1 if i > threshold else 0 for i in t]
    return tp

def f25(y, t):
    from sklearn.metrics import precision_score, recall_score
    import numpy as np
    p = max([precision_score(y, t), np.finfo(float).eps])
    r = max([recall_score(y, t), np.finfo(float).eps])
    return (1+0.25**2)*((p*r)/((0.25**2)*p+r))


def find_threshold(y, t, thresholds, metric):
    results = []
    for thresh in thresholds:
        tp = prediction_by_threshold(t, thresh)
        #tp = [1 if i > thresh else 0 for i in t]
        #print(tp)
        results.append(metric(y, tp))
    mi = results.index(max(results))
    return thresholds[mi]

def read_data(path, sample=None):
    import pandas as pd
    data = pd.read_csv(path)
    if sample:
        data = data.sample(sample)
    return data.values
    #return .sample(sample).values

if __name__ == "__main__":
    thresholds = [0.01 * i for i in range(1, 100)]
    threshold = find_threshold(y_test, t, thresholds, f25)
    tp = prediction_by_threshold(t, threshold)
    print(f25(y_test, tp))
