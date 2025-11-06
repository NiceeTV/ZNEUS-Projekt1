import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List
import wandb
import os

WANDB_PROJECT_NAME = "zneus-project-1"

EPOCH = 20
BATCH_SIZE = 256
LR = 0.001

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
            nn.Linear(input_size, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)


def evaluate(eloader: DataLoader, model, loss_fn):
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for X, y in eloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_test_loss += loss.item()
    return total_test_loss / len(eloader)


def train(train: CSVDataset, train_loader: DataLoader, eval_loader: DataLoader, loss_fn) -> (any, List[float], List[float]):
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
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        eval_losses.append(evaluate(eval_loader, model, loss_fn))
    
    return (model, train_losses, eval_losses)


def get_datasets(path: str) -> List[pd.DataFrame]:
    return [
        CSVDataset(f"{path}/train.csv"),
        CSVDataset(f"{path}/test.csv"),
        CSVDataset(f"{path}/eval.csv")
    ]


def _main(path: str):
    train_df, test_df, eval_df = get_datasets(path)
    train_loader = DataLoader(train_df, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_df, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_df, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.MSELoss()
    model, train_loss, eval_loss = train(train_df, train_loader, eval_loader, loss_fn)
    test_loss = evaluate(test_loader, model, loss_fn)

    return train_loss, eval_loss, test_loss

if __name__ == "__main__":
    path = "data/transformed/small"
    wandb.init(
        project=f"{WANDB_PROJECT_NAME}",
        name="small_first_try",
        config={
            "batch_size": BATCH_SIZE,
            "epoch": EPOCH,
            "lr": LR,
            "loss_fn": "MSELoss",
            "dataset_path": path
        }
    )
    
    tr_loss, ev_loss, te_loss = _main(path)
    for epoch, (tr_loss, ev_loss) in enumerate(zip(tr_loss, ev_loss), 1):
        wandb.log({
            "epoch": epoch,
            "train_loss": tr_loss,
            "eval_loss": ev_loss
        })
    wandb.log({"test_loss": te_loss})
    wandb.finish()