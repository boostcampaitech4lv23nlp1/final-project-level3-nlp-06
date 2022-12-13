from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, auc, roc_auc_score
import numpy as np
import torch


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
    
class Compute_metrics:
    def __init__(self, multi_label=False, num_labels=0):
        self.num_labels = num_labels
        self.multi_label = multi_label
        
    def compute_metrics(self, pred):
        if self.multi_label:
            sigmoid = torch.nn.Sigmoid()
            
            labels = pred.label_ids
            probs = sigmoid(torch.tensor(pred.predictions))
            preds = torch.zeros(probs.shape)
            preds[torch.where(probs >= 0.5)] = 1
            
            f1 = calc_f1_score(preds, labels)
            auprc = roc_auc_score(labels, preds, average="macro")
            acc = accuracy_score(labels, preds)
            
            return {
                "micro f1 score": f1,
                "auprc": auprc*100,
                "accuracy": acc
            }
        else:
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            probs = pred.predictions
            
            f1 = calc_f1_score(preds, labels)
            auprc_obj = Auprc(self.num_labels)
            auprc = auprc_obj.calc(probs, labels)
            acc = accuracy_score(labels, preds)
            
            return {
                "micro f1 score": f1,
                "auprc": auprc,
                "accuracy": acc
            }
            