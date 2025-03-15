import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# TPPModel: Post-processing model for fairness
class TPPModel(nn.Module):
    def __init__(self, input_dim):
        super(TPPModel, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, logits):
        return logits + self.fc(logits)

# Fairness loss function
def compute_fairness_loss(logits, sensitive_attr, labels):
    preds = logits.argmax(dim=-1)
    spd = demographic_parity_difference(labels, preds.cpu().numpy(), sensitive_attr)
    eod = equal_opportunity_difference(labels, preds.cpu().numpy(), sensitive_attr)
    classification_loss = nn.CrossEntropyLoss()(logits, torch.tensor(labels, dtype=torch.long).to(logits.device))
    return classification_loss + 0.1 * (spd + eod)

# Fairness metrics
def demographic_parity_difference(y_true, y_pred, sensitive_attr):
    groups = np.unique(sensitive_attr)
    group_rates = {}
    for group in groups:
        group_indices = np.where(sensitive_attr == group)[0]
        group_pred = y_pred[group_indices]
        group_rates[group] = np.mean(group_pred == 1)
    return abs(group_rates.get("female", 0) - group_rates.get("male", 0))

def equal_opportunity_difference(y_true, y_pred, sensitive_attr):
    groups = np.unique(sensitive_attr)
    group_tpr = {}
    for group in groups:
        group_indices = np.where((sensitive_attr == group) & (y_true == 1))[0]
        group_pred = y_pred[group_indices]
        group_tpr[group] = np.mean(group_pred == 1) if len(group_indices) > 0 else 0
    return abs(group_tpr.get("female", 0) - group_tpr.get("male", 0))

# Group accuracy function
def group_accuracy(y_true, y_pred, sensitive_attr, target_group):
    """Computes accuracy for a specific sensitive group."""
    group_indices = np.where(sensitive_attr == target_group)[0]
    return accuracy_score(y_true[group_indices], y_pred[group_indices])

# Main function
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data_df = pd.DataFrame(torch.load("content/train_logits.pt"))
    test_data_df = pd.DataFrame(torch.load("content/test_logits.pt"))
    train_sensitive_attr = np.load("content/train_sensitive_attr.npy")
    test_sensitive_attr = np.load("content/test_sensitive_attr.npy")
    train_y = np.load("content/train_labels.npy")
    test_y = np.load("content/test_labels.npy")

    # Initialize the TPP model
    tpp_model = TPPModel(input_dim=train_data_df.shape[1]).to(device)
    optimizer = optim.Adam(tpp_model.parameters(), lr=1e-3)
    epochs = 10

    # Training loop
    best_spd = float("inf")
    best_eod = float("inf")
    patience = 10  # Stop if no improvement for 10 epochs
    counter = 0

    for epoch in range(100):  # Run for max 100 epochs, but stop early
        tpp_model.train()
        optimizer.zero_grad()
        logits = torch.tensor(train_data_df.values, dtype=torch.float32).to(device)
        adjusted_logits = tpp_model(logits)
        loss = compute_fairness_loss(adjusted_logits, train_sensitive_attr, train_y)
        loss.backward()
        optimizer.step()

        # Compute fairness metrics
        train_preds = adjusted_logits.argmax(dim=-1).cpu().numpy()
        spd = demographic_parity_difference(train_y, train_preds, train_sensitive_attr)
        eod = equal_opportunity_difference(train_y, train_preds, train_sensitive_attr)

        print(f"Epoch {epoch + 1} - Loss: {loss.item()}, SPD: {spd}, EOD: {eod}")

        # Check for improvement
        if spd + eod < best_spd + best_eod:
            best_spd = spd
            best_eod = eod
            counter = 0  # Reset patience
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break  # Stop training when no improvement for 10 epochs

    # Evaluation
    tpp_model.eval()
    with torch.no_grad():
        test_logits = torch.tensor(test_data_df.values, dtype=torch.float32).to(device)
        adjusted_test_logits = tpp_model(test_logits)
        adjusted_preds = adjusted_test_logits.argmax(dim=-1).cpu().numpy()

        # Compute fairness metrics
        spd = demographic_parity_difference(test_y, adjusted_preds, test_sensitive_attr)
        eod = equal_opportunity_difference(test_y, adjusted_preds, test_sensitive_attr)

        # Compute overall accuracy and F1-score
        accuracy = accuracy_score(test_y, adjusted_preds)
        f1 = f1_score(test_y, adjusted_preds, average="macro")

        # Compute group-specific accuracy
        target_group_acc = group_accuracy(test_y, adjusted_preds, test_sensitive_attr, "female")
        non_target_group_acc = group_accuracy(test_y, adjusted_preds, test_sensitive_attr, "male")

        # Compute bias metric
        bias = abs(target_group_acc - non_target_group_acc)

        # Print results
        print(f"Adjusted Test Results - SPD: {spd}, EOD: {eod}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Target Group (Female) Accuracy: {target_group_acc:.4f}")
        print(f"Non-Target Group (Male) Accuracy: {non_target_group_acc:.4f}")
        print(f"Bias (|Target - Non-Target Accuracy|): {bias:.4f}")
        print(f"Test F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
