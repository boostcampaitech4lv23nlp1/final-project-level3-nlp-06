import pickle
import numpy as np


def get_prediction(result, label_map_dir, multi_label):
    if multi_label:
        with open(label_map_dir, "rb") as f:
            label_map = pickle.load(f)
        
        preds = []
        for pred in result.predictions:
            pred_label = []
            for i, p in enumerate(pred):
                if p > 0.5:
                    pred_label.append(label_map[i])
            preds.append(pred_label)
    else:
        preds = [np.argmax(pred) for pred in result.predictions]
        
    return preds
        