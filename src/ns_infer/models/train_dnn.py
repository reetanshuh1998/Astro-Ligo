import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class NSDataset(Dataset):
    """PyTorch Dataset for Neutron Star properties."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DeterministicMLP(nn.Module):
    """
    Standard deterministic Multi-Layer Perceptron for multi-output regression.
    """
    def __init__(self, input_dim=1, output_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
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
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def train_dnn(dataset_path, model_dir="data/models", epochs=100, batch_size=64, lr=0.003):
    """
    Trains a deterministic PyTorch MLP for multi-output regression (R, log10_Lambda).
    """
    print("Loading dataset for deterministic DNN training...")
    df = pd.read_csv(dataset_path)
    
    X_cols = ["M"]
    y_cols = ["R", "log10_Lambda"]
    
    # Load or reconstruct splits to match XGBoost exactly
    test_split_path = os.path.join(model_dir, "test_split.csv")
    if os.path.exists(test_split_path):
        print("  Reconstructing splits from pre-defined partitions...")
        test_df = pd.read_csv(test_split_path)
        # Remaining is train + val
        test_indices = set(test_df["P_c_cgs"])
        train_val_df = df[~df["P_c_cgs"].isin(test_indices)].reset_index(drop=True)
        
        # Split train_val into 88.2% train, 11.8% val (which is 75% and 10% of total)
        train_val_shuffled = train_val_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n_train = int(len(train_val_shuffled) * 0.882)
        train_df = train_val_shuffled.iloc[:n_train]
        val_df = train_val_shuffled.iloc[n_train:]
    else:
        # Fallback split
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(df_shuffled)
        n_train = int(n * 0.75)
        n_val = int(n * 0.10)
        train_df = df_shuffled.iloc[:n_train]
        val_df = df_shuffled.iloc[n_train:n_train+n_val]
        test_df = df_shuffled.iloc[n_train+n_val:]
        
    # Scale features and targets for robust neural network convergence
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(train_df[X_cols])
    y_train = scaler_y.fit_transform(train_df[y_cols])
    
    X_val = scaler_X.transform(val_df[X_cols])
    y_val = scaler_y.transform(val_df[y_cols])
    
    X_test = scaler_X.transform(test_df[X_cols])
    y_test = scaler_y.transform(test_df[y_cols])
    
    train_dataset = NSDataset(X_train, y_train)
    val_dataset = NSDataset(X_val, y_val)
    test_dataset = NSDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model construction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeterministicMLP(input_dim=1, output_dim=2).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float("inf")
    best_weights = None
    
    print("Beginning DNN training loops...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step()
        
        # Save best model weight dictionary
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
    # Load best weights
    model.load_state_dict(best_weights)
    
    # Save scalers and model
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "dnn_scaler_X.pkl"), "wb") as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(model_dir, "dnn_scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)
        
    torch.save(model.state_dict(), os.path.join(model_dir, "dnn_model.pt"))
    print("Deterministic DNN training completed.")
    return model

if __name__ == "__main__":
    train_dnn("data/processed/ns_dataset_test.csv", epochs=10)
