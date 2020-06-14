import re


def estimate_model(labels, prediction):
    """

    :param labels: vector of floats - the true labels
    :param prediction: vector of floats - model's predictions.
    """
    TP, TN, FP, FN = 0, 0, 0, 0
    for l, p in zip(labels, prediction):
        if l == 1:
            if p == 1:
                TP += 1
            else:
                FN += 1
        else:
            if p == 1:
                FP += 1
            else:
                TN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = (TP / (TP + FN)) if TP + FN != 0 else 0
    precision = (TP / (TP + FP)) if TP + FP != 0 else 0
    beta = 0.25
    F25 = ((1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)) \
        if beta ** 2 * precision + recall != 0 else 0
    print(f"accuracy: {accuracy}, recall: {recall}, precision: {precision}, F0.25: {F25}")


def validation():
    """
    estimate the model based on the validation.txt file.
    """
    # calc: -0.0, label: 1.0
    o = re.compile(r"calc: ([0-9.\-]*), label: ([0-1.]*)")
    calc, labels = [], []
    with open("algorithms/validation.txt", 'r') as f:
        for line in f:
            result = o.match(line)
            if result:
                c, l = float(result.group(1)), float(result.group(2))
                calc.append(c)
                labels.append(l)

    # todo: change threshold to maximize precision
    # apply threshold
    predictions = [1 if c > 0.5 else 0 for c in calc]
    estimate_model(labels, predictions)


