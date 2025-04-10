import tensorflow as tf
import numpy as np
from datasets import Dataset

## funker om y_pred är sorterad
def mean_reciprocal_rank(y_true, y_pred, k):
    all_rr = []
    for i in range(len(y_true)):
        pred = np.array(y_pred[i], dtype='float32')
        true = np.array(y_true[i], dtype='float32')
        data_dict = {"pred" : pred, "true" : true}
        ds = Dataset.from_dict(data_dict)
        ds_sorted = ds.sort("pred", reverse=True)

        pred = ds_sorted["pred"]
        true = ds_sorted["true"]
        print("y_pred", pred)
        print("y_true", true)

        true_k = true[:k]
        try:
            rr = 1 / (true_k.index(1)+1)
        except ValueError:
            rr = 0
        all_rr.append(rr)

    mean_rr = sum(all_rr)/(i+1)
    return mean_rr

# Example usage
y_true = [[0, 0, 1, 0, 0]]
y_pred = [[5, 3, 1, 2, 4]]
mrr = mean_reciprocal_rank(y_true, y_pred, k=3)
print(f"MRR: {mrr}")