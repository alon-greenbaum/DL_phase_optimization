import numpy as np
import matplotlib.pyplot as plt
import os

def compute_and_log_metrics(gt_img, cnn_img, out_dir, name):
    """
    Computes and logs performance metrics for the full image and its top, middle, and bottom thirds.

    Args:
        gt_img (np.ndarray): The ground truth image.
        cnn_img (np.ndarray): The CNN inference image.
    """
    h = gt_img.shape[0]  # Assuming the first dimension is height

    # Full image
    gt_full = gt_img
    cnn_full = cnn_img
    precision_full, recall_full, f1_full = compute_metrics(gt_full, cnn_full)
    print(f"Full image: Precision: {precision_full:.4f}, Recall: {recall_full:.4f}, F1: {f1_full:.4f}")

    # Top 1/3
    gt_top = gt_img[:h//3, :]
    cnn_top = cnn_img[:h//3, :]
    precision_top, recall_top, f1_top = compute_metrics(gt_top, cnn_top)
    print(f"Top 1/3: Precision: {precision_top:.4f}, Recall: {recall_top:.4f}, F1: {f1_top:.4f}")

    # Middle 1/3
    gt_middle = gt_img[h//3:2*h//3, :]
    cnn_middle = cnn_img[h//3:2*h//3, :]
    precision_middle, recall_middle, f1_middle = compute_metrics(gt_middle, cnn_middle)
    print(f"Middle 1/3: Precision: {precision_middle:.4f}, Recall: {recall_middle:.4f}, F1: {f1_middle:.4f}")

    # Bottom 1/3
    gt_bottom = gt_img[2*h//3:, :]
    cnn_bottom = cnn_img[2*h//3:, :]
    precision_bottom, recall_bottom, f1_bottom = compute_metrics(gt_bottom, cnn_bottom)
    print(f"Bottom 1/3: Precision: {precision_bottom:.4f}, Recall: {recall_bottom:.4f}, F1: {f1_bottom:.4f}")

    metrics_out_path = os.path.join(out_dir, "metrics.txt")
    with open(metrics_out_path, "a") as f:
        f.write(f"Metrics for {name}:\n")
        f.write(f"  Full image: Precision: {precision_full:.4f}, Recall: {recall_full:.4f}, F1: {f1_full:.4f}\n")
        f.write(f"  Top 1/3: Precision: {precision_top:.4f}, Recall: {recall_top:.4f}, F1: {f1_top:.4f}\n")
        f.write(f"  Middle 1/3: Precision: {precision_middle:.4f}, Recall: {recall_middle:.4f}, F1: {f1_middle:.4f}\n")
        f.write(f"  Bottom 1/3: Precision: {precision_bottom:.4f}, Recall: {recall_bottom:.4f}, F1: {f1_bottom:.4f}\n")
        
    

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

def save_heatmap(image, out_dir, filename, steps=10):
    """
    Saves a heatmap of the input image with specified range and steps.

    Args:
        image (np.ndarray): The input image (2D array).
        out_dir (str): The directory to save the heatmap.
        filename (str): The base filename for the saved heatmap image.
        steps (int): The number of steps in the range (default: 10).
    """
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Define the range and steps
    vmin = 0
    vmax = 1
    step_size = (vmax - vmin) / steps

    # Create the heatmap
    plt.imshow(image, cmap='magma', vmin=vmin, vmax=vmax)
    plt.colorbar(ticks=np.arange(vmin, vmax + step_size, step_size))
    plt.title("Heatmap")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save the heatmap
    filepath = os.path.join(out_dir, f"{filename}_heatmap.png")
    plt.savefig(filepath)
    plt.close()  # Close the figure to release memory
    print(f"Heatmap saved to {filepath}")
