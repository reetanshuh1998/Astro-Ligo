import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from ns_infer.models.train_dnn import NSDataset

class EnsembleGaussianMLP(nn.Module):
    """
    Dense neural network predicting both mean and variance for multi-output regression.
    Uses MC Dropout for epistemic uncertainty sampling.
    """
    def __init__(self, input_dim=1, num_targets=2, hidden_dim=128):
        super().__init__()
        self.num_targets = num_targets
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Multi-head output for mean and log-variance
        self.out_mean = nn.Linear(hidden_dim // 2, num_targets)
        self.out_logvar = nn.Linear(hidden_dim // 2, num_targets)
        
    def forward(self, x):
        features = self.shared(x)
        mean = self.out_mean(features)
        logvar = self.out_logvar(features)
        
        # Clamp log-variance to prevent numerical exponential overflow/underflow
        logvar = torch.clamp(logvar, min=-8.0, max=4.0)
        return mean, logvar

class GaussianNLLLoss(nn.Module):
    """Heteroscedastic Gaussian Negative Log-Likelihood loss function."""
    def __init__(self):
        super().__init__()
        
    def forward(self, mean, logvar, targets):
        # Loss = 0.5 * (exp(-logvar) * (targets - mean)^2 + logvar)
        inv_var = torch.exp(-logvar)
        sq_err = (targets - mean) ** 2
        nll = 0.5 * (inv_var * sq_err + logvar)
        return torch.mean(nll)

def train_ensemble_member(member_id, train_loader, val_loader, device, epochs, lr, model_dir):
    """Trains a single Gaussian neural network member of the deep ensemble."""
    print(f"  Training Ensemble Member {member_id}...")
    model = EnsembleGaussianMLP(input_dim=1, num_targets=2).to(device)
    
    criterion = GaussianNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float("inf")
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            mean, logvar = model(inputs)
            loss = criterion(mean, logvar, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                mean, logvar = model(inputs)
                loss = criterion(mean, logvar, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            
    model.load_state_dict(best_weights)
    model_path = os.path.join(model_dir, f"bnn_member_{member_id}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  Ensemble Member {member_id} trained. Saved to {model_path}")
    return model

def train_bnn(dataset_path, model_dir="data/models", num_members=5, epochs=100, batch_size=64, lr=0.003):
    """
    Trains a Gaussian Deep Ensemble (5 members) with heteroscedastic NLL loss.
    """
    print(f"Loading dataset for BNN Ensemble training ({num_members} members)...")
    df = pd.read_csv(dataset_path)
    
    X_cols = ["M"]
    y_cols = ["R", "log10_Lambda"]
    
    # Reconstruct same splits
    test_split_path = os.path.join(model_dir, "test_split.csv")
    if os.path.exists(test_split_path):
        test_df = pd.read_csv(test_split_path)
        test_indices = set(test_df["P_c_cgs"])
        train_val_df = df[~df["P_c_cgs"].isin(test_indices)].reset_index(drop=True)
        
        # Shuffled split
        train_val_shuffled = train_val_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n_train = int(len(train_val_shuffled) * 0.882)
        train_df = train_val_shuffled.iloc[:n_train]
        val_df = train_val_shuffled.iloc[n_train:]
    else:
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(df_shuffled)
        n_train = int(n * 0.75)
        n_val = int(n * 0.10)
        train_df = df_shuffled.iloc[:n_train]
        val_df = df_shuffled.iloc[n_train:n_train+n_val]
        test_df = df_shuffled.iloc[n_train+n_val:]
        
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(train_df[X_cols])
    y_train = scaler_y.fit_transform(train_df[y_cols])
    
    X_val = scaler_X.transform(val_df[X_cols])
    y_val = scaler_y.transform(val_df[y_cols])
    
    X_test = scaler_X.transform(test_df[X_cols])
    y_test = scaler_y.transform(test_df[y_cols])
    
    # Save scalers for downstream ensembled predictions
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "bnn_scaler_X.pkl"), "wb") as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(model_dir, "bnn_scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)
        
    train_dataset = NSDataset(X_train, y_train)
    val_dataset = NSDataset(X_val, y_val)
    
    # Enable different seeds for each member initializations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ensemble = []
    for i in range(num_members):
        # We vary the dataloader shuffling seed dynamically per member
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        member = train_ensemble_member(i, train_loader, val_loader, device, epochs, lr, model_dir)
        ensemble.append(member)
        
    print("Gaussian Deep Ensemble training successfully completed.")
    return ensemble

def predict_ensemble(ensemble, X_input, scaler_X, scaler_y, device, mc_runs=20):
    """
    Performs inference over the trained ensemble with MC Dropout active.
    Extracts prediction means and decoupled aleatoric, epistemic, and combined uncertainties.
    """
    X_scaled = scaler_X.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    means = []
    vars_aleatoric = []
    
    # For MC Dropout, we want to leave the dropout layers ACTIVE during evaluation.
    # To do this in PyTorch, we can call .eval() but then force the shared dropout layers
    # to remain in training mode.
    for member in ensemble:
        member.eval()
        # Force dropout layers back to training mode
        for m in member.modules():
            if isinstance(m, nn.Dropout):
                m.train()
                
        member_means = []
        member_vars = []
        
        with torch.no_grad():
            for _ in range(mc_runs):
                pred_mean, pred_logvar = member(X_tensor)
                
                # Inverse transform to get back physical values
                # We need to un-scale the mean and variance.
                # Scaler: y_scaled = (y - mean_y) / scale_y => y = y_scaled * scale_y + mean_y
                # Variance: var_physical = var_scaled * scale_y^2
                mean_phys = pred_mean.cpu().numpy() * scaler_y.scale_ + scaler_y.mean_
                var_phys = np.exp(pred_logvar.cpu().numpy()) * (scaler_y.scale_ ** 2)
                
                member_means.append(mean_phys)
                member_vars.append(var_phys)
                
        # Stack runs for this member: shape (mc_runs, N, targets)
        member_means = np.stack(member_means)
        member_vars = np.stack(member_vars)
        
        # Store results
        means.append(member_means)
        vars_aleatoric.append(member_vars)
        
    # Concatenate all members: shape (num_members * mc_runs, N, targets)
    # Total samples = 5 * 20 = 100 predictions
    means = np.concatenate(means, axis=0)
    vars_aleatoric = np.concatenate(vars_aleatoric, axis=0)
    
    # Combined predictive mean
    pred_mean_comb = np.mean(means, axis=0)
    
    # Decouple uncertainties
    # 1. Aleatoric uncertainty (intrinsic data noise)
    aleatoric_var = np.mean(vars_aleatoric, axis=0)
    
    # 2. Epistemic uncertainty (model parameter/functional uncertainty)
    epistemic_var = np.var(means, axis=0)
    
    # 3. Combined total variance
    total_var = aleatoric_var + epistemic_var
    total_std = np.sqrt(total_var)
    
    return pred_mean_comb, total_std, np.sqrt(aleatoric_var), np.sqrt(epistemic_var)

if __name__ == "__main__":
    train_bnn("data/processed/ns_dataset_test.csv", epochs=10)
