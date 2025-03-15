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

# Improved fairness adjustment model with a small neural network
class FairnessAdjustmentModel(nn.Module):
    def __init__(self, input_dim):
        super(FairnessAdjustmentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.fc2 = nn.Linear(input_dim * 2, input_dim)
        self.activation = nn.ReLU()
    
    def forward(self, logits):
        return logits + self.fc2(self.activation(self.fc1(logits)))

# Differentiable fairness loss
class FairnessLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(FairnessLoss, self).__init__()
        self.alpha = alpha  # Fairness regularization weight
        self.criterion = nn.CrossEntropyLoss()
    
    def demographic_parity_loss(self, logits, sensitive_attr):
        preds = logits.argmax(dim=-1)
        groups = torch.unique(sensitive_attr)
        group_rates = {}
        
        for group in groups:
            group_indices = (sensitive_attr == group).nonzero(as_tuple=True)[0]
            group_preds = preds[group_indices]
            group_rates[group.item()] = torch.mean((group_preds == 1).float())
        
        group_values = list(group_rates.values())
        return torch.abs(group_values[0] - group_values[1]) if len(group_values) == 2 else 0
    
    def equal_opportunity_loss(self, logits, sensitive_attr, labels):
        preds = logits.argmax(dim=-1)
        groups = torch.unique(sensitive_attr)
        group_tpr = {}
        
        for group in groups:
            group_indices = ((sensitive_attr == group) & (labels == 1)).nonzero(as_tuple=True)[0]
            if len(group_indices) > 0:
                group_preds = preds[group_indices]
                group_tpr[group.item()] = torch.mean((group_preds == 1).float())
            else:
                group_tpr[group.item()] = torch.tensor(0.0)
        
        group_values = list(group_tpr.values())
        return torch.abs(group_values[0] - group_values[1]) if len(group_values) == 2 else 0
    
    def forward(self, logits, labels, sensitive_attr):
        classification_loss = self.criterion(logits, labels)
        spd_loss = self.demographic_parity_loss(logits, sensitive_attr)
        eod_loss = self.equal_opportunity_loss(logits, sensitive_attr, labels)
        return classification_loss + self.alpha * (spd_loss + eod_loss)

# Training function
def train_fairness_model(train_data, train_sensitive_attr, train_labels, input_dim, device, epochs=100, patience=10):
    model = FairnessAdjustmentModel(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = FairnessLoss(alpha=0.1).to(device)
    
    # Convert categorical sensitive attributes to numerical
    unique_groups = np.unique(train_sensitive_attr)
    group_mapping = {group: i for i, group in enumerate(unique_groups)}
    train_sensitive_attr = np.array([group_mapping[group] for group in train_sensitive_attr])
    
    best_spd = float("inf")
    best_eod = float("inf")
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = torch.tensor(train_data.values, dtype=torch.float32).to(device)
        sensitive_attr = torch.tensor(train_sensitive_attr, dtype=torch.long).to(device)
        labels = torch.tensor(train_labels, dtype=torch.long).to(device)
        
        adjusted_logits = model(logits)
        loss = criterion(adjusted_logits, labels, sensitive_attr)
        loss.backward()
        optimizer.step()
        
        train_preds = adjusted_logits.argmax(dim=-1).cpu().numpy()
        spd = criterion.demographic_parity_loss(adjusted_logits, sensitive_attr).item()
        eod = criterion.equal_opportunity_loss(adjusted_logits, sensitive_attr, labels).item()
        
        print(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}, SPD: {spd:.4f}, EOD: {eod:.4f}")
        
        if spd + eod < best_spd + best_eod:
            best_spd, best_eod = spd, eod
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break
    
    return model, group_mapping

# Main execution
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data_df = pd.DataFrame(torch.load("content/train_logits.pt"))
    test_data_df = pd.DataFrame(torch.load("content/test_logits.pt"))
    train_sensitive_attr = np.load("content/train_sensitive_attr.npy")
    test_sensitive_attr = np.load("content/test_sensitive_attr.npy")
    train_y = np.load("content/train_labels.npy")
    test_y = np.load("content/test_labels.npy")
    
    model, group_mapping = train_fairness_model(train_data_df, train_sensitive_attr, train_y, train_data_df.shape[1], device)
    evaluate_model(model, test_data_df, test_sensitive_attr, test_y, device, group_mapping)

if __name__ == "__main__":
    main()
