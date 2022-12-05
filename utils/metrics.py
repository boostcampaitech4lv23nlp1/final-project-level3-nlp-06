from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, auc
import numpy as np


def calc_f1_score(preds, labels):
    return f1_score(labels, preds, average="micro") * 100.0

class Auprc:
    def __init__(self, num_labels: int):
        self.num_labels = num_labels
        
    def calc(self, probs, labels):
        labels = np.eye(self.num_labels)[labels]
        score = np.zeros((self.num_labels,))
        
        for c in range(self.num_labels):
            targets_c = labels.take([c], axis=1).ravel()
            preds_c = probs.take([c], axis=1).ravel()
            precision, recall, _ = precision_recall_curve(targets_c, preds_c)
            score[c] = auc(recall, precision)
        return np.average(score) * 100.0


