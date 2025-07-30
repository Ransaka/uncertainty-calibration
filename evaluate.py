import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def compute_ece(confidences, predictions, labels, n_bins=15):
    """
    Computes the Expected Calibration Error (ECE).
    
    Args:
        confidences (np.array): 1D array of confidence scores.
        predictions (np.array): 1D array of predictions (0 or 1).
        labels (np.array): 1D array of true labels (0 or 1).
        n_bins (int): Number of bins to use for ECE calculation.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def get_all_pixel_data(uncertainty_dir):
    """Loads all saved .npz files into flat numpy arrays."""
    all_pred_var = []
    all_sal_var = []
    all_mean_pred = []
    all_true_mask = []

    for fname in tqdm(os.listdir(uncertainty_dir), desc="Loading maps"):
        if fname.endswith('.npz'):
            data = np.load(os.path.join(uncertainty_dir, fname))
            all_pred_var.append(data['pred_variance'].flatten())
            all_sal_var.append(data['saliency_variance'].flatten())
            all_mean_pred.append(data['mean_prediction'].flatten())
            all_true_mask.append(data['true_mask'].flatten())

    return (
        np.concatenate(all_pred_var),
        np.concatenate(all_sal_var),
        np.concatenate(all_mean_pred),
        np.concatenate(all_true_mask)
    )

def tune_lambda_and_evaluate(val_dir, test_dir):
    """
    FINAL CORRECTED VERSION: Handles potential NaN values from saliency variance.
    """
    # 1. Load validation data
    val_pred_var, val_sal_var, val_mean_pred, val_true_mask = get_all_pixel_data(val_dir)
    val_foreground_mask = val_true_mask.astype(bool)
    
    # 2. Get normalization constants, SAFELY ignoring potential NaNs
    pred_min, pred_max = np.nanmin(val_pred_var), np.nanmax(val_pred_var)
    sal_min, sal_max = np.nanmin(val_sal_var), np.nanmax(val_sal_var)

    def normalize_and_sanitize(data, d_min, d_max):
        # Perform normalization
        normalized_data = (data - d_min) / (d_max - d_min + 1e-8)
        # Replace any NaNs that might have been in the input or created by division
        np.nan_to_num(normalized_data, copy=False, nan=0.0)
        return normalized_data

    # Normalize the full data arrays
    norm_val_pred_var = normalize_and_sanitize(val_pred_var, pred_min, pred_max)
    norm_val_sal_var = normalize_and_sanitize(val_sal_var, sal_min, sal_max)
    
    # 3. Tune lambda on the validation set's foreground
    best_lambda, best_ece = -1, float('inf')
    
    val_pixel_preds = (val_mean_pred > 0.5).astype(int)
    val_labels_fg = val_true_mask.astype(int)[val_foreground_mask]
    val_preds_fg = val_pixel_preds[val_foreground_mask]

    print("\n--- Tuning Lambda on FOREGROUND Validation Set ---")
    for lam in np.linspace(0, 1, 11):
        u_joint = lam * norm_val_pred_var + (1 - lam) * norm_val_sal_var
        confidences = 1 - u_joint
        confidences_fg = confidences[val_foreground_mask]

        ece = compute_ece(confidences_fg, val_preds_fg, val_labels_fg)
        print(f"Lambda = {lam:.1f}, ECE = {ece:.5f}")
        
        if ece < best_ece:
            best_ece = ece
            best_lambda = lam

    print(f"\nBest Lambda found: {best_lambda:.1f} with Foreground ECE: {best_ece:.5f}\n")

    # 4. Final Evaluation on the Test Set
    print("--- Final Evaluation on FOREGROUND Test Set ---")
    test_pred_var, test_sal_var, test_mean_pred, test_true_mask = get_all_pixel_data(test_dir)
    test_foreground_mask = test_true_mask.astype(bool)
    
    norm_test_pred_var = normalize_and_sanitize(test_pred_var, pred_min, pred_max)
    norm_test_sal_var = normalize_and_sanitize(test_sal_var, sal_min, sal_max)
    
    test_pixel_preds = (test_mean_pred > 0.5).astype(int)
    test_labels_fg = test_true_mask.astype(int)[test_foreground_mask]
    test_preds_fg = test_pixel_preds[test_foreground_mask]

    # Baseline 1: Predictive Variance Only
    conf_pred_only_fg = (1 - norm_test_pred_var)[test_foreground_mask]
    ece_pred_only = compute_ece(conf_pred_only_fg, test_preds_fg, test_labels_fg)
    print(f"ECE (Predictive Variance Only, lambda=1.0): {ece_pred_only:.5f}")

    # Baseline 2: Saliency Variance Only
    conf_sal_only_fg = (1 - norm_test_sal_var)[test_foreground_mask]
    ece_sal_only = compute_ece(conf_sal_only_fg, test_preds_fg, test_labels_fg)
    print(f"ECE (Saliency Variance Only, lambda=0.0): {ece_sal_only:.5f}")

    # Proposed Method: Joint Score
    u_joint_best = best_lambda * norm_test_pred_var + (1 - best_lambda) * norm_test_sal_var
    conf_joint_best_fg = (1 - u_joint_best)[test_foreground_mask]
    ece_joint_best = compute_ece(conf_joint_best_fg, test_preds_fg, test_labels_fg)
    print(f"ECE (Joint Score, best lambda={best_lambda:.1f}): {ece_joint_best:.5f}")
