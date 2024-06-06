import time


def print_or_output(string, d, out_file):
    if d == "pc":
        print(string)
    elif d == "cluster":
        print(string)
        with open(out_file, "a") as f:
            f.write(string + "\n")


def confusion_matrix(prediction, labels, threshold=0.5):
    """
        By @MelissaLP
        This function computes the confusion matrix and returns TP, TN, FP, FN counts.
    Input
    -----

    prediction: probability vector of predicted classes
    labels: class labels
    threshold: decision threshold

    """
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    # print(prediction[0])
    for i in range(len(prediction)):
        if prediction[i] > threshold:
            if labels[i] == 1:
                tp = tp + 1
            if labels[i] == 0:
                fp = fp + 1
        if prediction[i] < threshold:
            if labels[i] == 1:
                fn = fn + 1
            if labels[i] == 0:
                tn = tn + 1

    return tp, tn, fp, fn


if __name__ == "__main__":
    l = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in l:
        print(
            confusion_matrix(
                [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9],
                [0, 0, 0, 1, 1, 1, 1, 1],
                threshold=threshold,
            )
        )
