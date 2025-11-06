import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List
import wandb
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

WANDB_PROJECT_NAME = "zneus-project-1"
IS_WANDB = False

EPOCH = 128
BATCH_SIZE = 512
LR = 0.004

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device}")

class CSVDataset(Dataset):
    def __init__(self, path: str):
        df = pd.read_csv(path)
        self.X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).view(-1, 1) # last element is predicted value

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleNN(nn.Module):
    def __init__(self, input_size: int):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)


def get_inverse_transformed(y, y_transformer=None):
    if y_transformer is not None:
        y = y.cpu().numpy()
        y = y_transformer(y)
        return torch.tensor(y, dtype=torch.float32, device=device)
    return y

def evaluate(eloader: DataLoader, model, loss_fn, y_transformer=None):
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for X, y in eloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred = get_inverse_transformed(y_pred, y_transformer)
            y = get_inverse_transformed(y, y_transformer)
            total_test_loss += loss_fn(y_pred, y).item()
    return total_test_loss / len(eloader)


def train(train: CSVDataset, train_loader: DataLoader, eval_loader: DataLoader, loss_fn, reverse_transform=None) -> (any, List[float], List[float]):
    model = SimpleNN(train.X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_losses = []
    eval_losses = []

    for epoch in range(EPOCH):
        model.train()
        total_train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            # calculate non-normalized loss, for logging
            with torch.no_grad():
                y_orig = get_inverse_transformed(y.detach(), reverse_transform)
                y_pred_orig = get_inverse_transformed(y_pred.detach(), reverse_transform)
                total_train_loss += loss_fn(y_pred_orig, y_orig).item()
        avg_train_loss = total_train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        eval_losses.append(evaluate(eval_loader, model, loss_fn, reverse_transform))
    
    return (model, train_losses, eval_losses)


def get_datasets(path: str) -> List[pd.DataFrame]:
    return [
        CSVDataset(f"{path}/train.csv"),
        CSVDataset(f"{path}/test.csv"),
        CSVDataset(f"{path}/eval.csv")
    ]


def _main(path: str, reverse_transform):
    train_df, test_df, eval_df = get_datasets(path)
    train_loader = DataLoader(train_df, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_df, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_df, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.MSELoss()
    model, train_loss, eval_loss = train(train_df, train_loader, eval_loader, loss_fn, reverse_transform)
    test_loss = evaluate(test_loader, model, loss_fn, reverse_transform)

    return train_loss, eval_loss, test_loss

if __name__ == "__main__":
    path = "data/transformed/full_features"
    if IS_WANDB:
        wandb.init(
            project=f"{WANDB_PROJECT_NAME}",
            name="full_norm_back_try",
            config={
                "batch_size": BATCH_SIZE,
                "epoch": EPOCH,
                "lr": LR,
                "loss_fn": "MSELoss",
                "dataset_path": path
            }
        )
    
    target_transformer = joblib.load(f"{path}/house_value_scaler.pkl")
    def reverse_transform(val):
        return target_transformer.inverse_transform(val)
    tr_loss, ev_loss, te_loss = _main(path, reverse_transform)
    
    if IS_WANDB:
        for epoch, (tr_loss, ev_loss) in enumerate(zip(tr_loss, ev_loss), 1):
            wandb.log({
                "epoch": epoch,
                "train_loss": tr_loss,
                "eval_loss": ev_loss
            })
        wandb.log({"test_loss": te_loss})
        wandb.finish()
    
    print("Training loss: ", te_loss)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(tr_loss) + 1), tr_loss, label="Training Loss", linewidth=2)
    plt.plot(range(1, len(ev_loss) + 1), ev_loss, label="Evaluation Loss", linewidth=2)
    plt.title("Training and Evaluation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
