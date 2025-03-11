import numpy as np

def compute_metrics(y_true, y_pred, threshold=0.5):
    # Binarize predictions and ground truth
    y_true_bin = (y_true > 0.5).astype(np.uint8)
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    TP = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    FP = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    FN = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1
