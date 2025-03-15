import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load saved logits and labels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_logits = torch.tensor(pd.DataFrame(torch.load("content/test_logits.pt")).values, dtype=torch.float32).to(device)
test_preds = test_logits.argmax(dim=-1).cpu().numpy()
test_y = np.load("content/test_labels.npy")  # Ground truth labels
test_sensitive_attr = np.load("content/test_sensitive_attr.npy")  # Sensitive attribute

# Fairness metric functions
def demographic_parity_difference(y_true, y_pred, sensitive_attr):
    groups = np.unique(sensitive_attr)
    group_rates = {group: np.mean(y_pred[np.where(sensitive_attr == group)[0]] == 1) for group in groups}
    return abs(group_rates.get("female", 0) - group_rates.get("male", 0))

def equal_opportunity_difference(y_true, y_pred, sensitive_attr):
    groups = np.unique(sensitive_attr)
    group_tpr = {
        group: np.mean(y_pred[np.where((sensitive_attr == group) & (y_true == 1))[0]] == 1)
        if len(np.where((sensitive_attr == group) & (y_true == 1))[0]) > 0 else 0
        for group in groups
    }
    return abs(group_tpr.get("female", 0) - group_tpr.get("male", 0))

# Compute fairness metrics
spd = demographic_parity_difference(test_y, test_preds, test_sensitive_attr)
eod = equal_opportunity_difference(test_y, test_preds, test_sensitive_attr)

# Compute overall accuracy
accuracy = accuracy_score(test_y, test_preds)
f1 = f1_score(test_y, test_preds, average='weighted')

# Compute accuracy for each group
def group_accuracy(y_true, y_pred, sensitive_attr, target_group):
    """Computes accuracy for a specific sensitive group."""
    group_indices = np.where(sensitive_attr == target_group)[0]
    return accuracy_score(y_true[group_indices], y_pred[group_indices])

# Calculate accuracy for target (female) and non-target (male) groups
target_group_acc = group_accuracy(test_y, test_preds, test_sensitive_attr, "female")
non_target_group_acc = group_accuracy(test_y, test_preds, test_sensitive_attr, "male")

# Compute bias metric
bias = abs(target_group_acc - non_target_group_acc)

# Print results
print(f"Base Model Results:")
print(f"Statistical Parity Difference (SPD): {spd}")
print(f"Equal Opportunity Difference (EOD): {eod}")
print(f"Overall Accuracy: {accuracy}")
print(f"Target Group (Female) Accuracy: {target_group_acc}")
print(f"Non-Target Group (Male) Accuracy: {non_target_group_acc}")
print(f"Bias (|Target - Non-Target Accuracy|): {bias}")
print(f"F1 Score: {f1}")

# Run post-processing directly in Python
from postproc_fairness.fairmain import main
main()
