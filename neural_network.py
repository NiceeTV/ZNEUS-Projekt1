import itertools

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
IS_WANDB = True
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

EPOCH = 50
BATCH_SIZE = 128
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


class SimpleNN_dropout(nn.Module):
    def __init__(self, input_size: int):
        super(SimpleNN_dropout, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class SimpleNN_batch_norm(nn.Module):
    def __init__(self, input_size: int):
        super(SimpleNN_batch_norm, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class SimpleNN_Res(nn.Module): #skip connections
    def __init__(self, input_size: int):
        super(SimpleNN_Res, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.act1 = nn.LeakyReLU(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.LeakyReLU(0.1)

        self.fc3 = nn.Linear(64, 32)
        self.act3 = nn.LeakyReLU(0.1)

        self.out = nn.Linear(32, 1)

        #projection for skip connections
        self.skip_proj = nn.Linear(input_size, 64)

    def forward(self, x):
        #first layer
        x1 = self.act1(self.fc1(x))

        #skip connection: input x + projection to 64
        skip = self.skip_proj(x)
        x2 = self.act2(self.fc2(x1) + skip) #residual connection

        #next layer
        x3 = self.act3(self.fc3(x2))

        out = self.out(x3)
        return out


class SimpleNN_Bottleneck(nn.Module):
    def __init__(self, input_size: int, bottleneck_size: int = 16):
        super(SimpleNN_Bottleneck, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.act1 = nn.LeakyReLU(0.3)

        #bottleneck layer
        self.fc2 = nn.Linear(128, bottleneck_size)
        self.act2 = nn.LeakyReLU(0.1)

        #output layer
        self.fc3 = nn.Linear(bottleneck_size, 32)
        self.act3 = nn.LeakyReLU(0.1)

        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x1 = self.act1(self.fc1(x))
        x2 = self.act2(self.fc2(x1))
        x3 = self.act3(self.fc3(x2))
        out = self.out(x3)
        return out


def get_inverse_transformed(y, y_transformer=None):
    if y_transformer is not None:
        y = y.cpu().numpy().reshape(-1, 1)
        y = y_transformer.inverse_transform(y)
        return torch.tensor(y, dtype=torch.float32, device=device)
    return y


def evaluate(eloader: DataLoader, model, loss_fn, y_transformer=None, is_test=False):
    model.eval()
    total_test_loss = 0
    total_test_loss_original = 0
    num_of_batches = len(eloader)

    mse_losses, rmse_losses = [], [] #for test only

    with torch.no_grad():
        if is_test:
            print("\n====TESTING====")

        for index, (X, y) in enumerate(eloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss_transformed = loss_fn(y_pred, y).item()
            mse_losses.append(loss_transformed)
            total_test_loss += loss_transformed

            y_pred = get_inverse_transformed(y_pred, y_transformer)
            y = get_inverse_transformed(y, y_transformer)

            loss_original = loss_fn(y_pred, y).item()
            total_test_loss_original += loss_original
            if is_test:
                rmse = loss_original ** 0.5
                rmse_losses.append(rmse)

                print(f"Test eval: Batch {index + 1:03d}: MSE={loss_transformed:.4f}, RMSE={rmse:.4f}")

    return total_test_loss/num_of_batches, total_test_loss_original/num_of_batches, mse_losses, rmse_losses, num_of_batches


def train(train: CSVDataset, train_loader: DataLoader, eval_loader: DataLoader, loss_fn, transformer=None, model=None, optimizer=None) -> (any, List[float], List[float]):
    model = model if model else SimpleNN_Bottleneck(train.X.shape[1]).to(device) #in tuning we pass the model, else its base

    optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=LR) #same with optimizer
    train_mse, eval_mse = [], []
    train_rmse, eval_rmse = [], []

    print("\n====TRAINING====")
    for epoch in range(EPOCH):
        model.train()
        total_train_mse = 0
        total_train_rmse = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            #predict y
            y_pred = model(X)

            #calculate loss
            loss = loss_fn(y_pred, y)
            total_train_mse += loss.item()

            #backward pass
            loss.backward()
            optimizer.step()
            
            #calculate non-normalized loss, for logging
            with torch.no_grad():
                #inverse transform predicted and true y values
                y_orig = get_inverse_transformed(y.detach(), transformer)
                y_pred_orig = get_inverse_transformed(y_pred.detach(), transformer)

                #calculate mse and rmse from results
                mse = loss_fn(y_pred_orig, y_orig).item()
                rmse = mse ** 0.5

                total_train_rmse += rmse

        #average train mse/rmse per epoch
        avg_train_mse = total_train_mse / len(train_loader)
        avg_train_rmse = total_train_rmse / len(train_loader)

        #test on eval set
        eval_mse_e, eval_mse_original, _, _, _ = evaluate(eval_loader, model, loss_fn, transformer)
        eval_rmse_e = eval_mse_original ** 0.5

        #add to arrays
        train_mse.append(avg_train_mse)
        eval_mse.append(eval_mse_e)
        train_rmse.append(avg_train_rmse)
        eval_rmse.append(eval_rmse_e)

        print(f"Epoch {epoch+1:03d}: train RMSE={avg_train_rmse:.4f}, eval RMSE={eval_rmse_e:.4f}")
    
    return model, train_mse, eval_mse, train_rmse, eval_rmse


def get_datasets(path: str) -> List[CSVDataset]:
    return [
        CSVDataset(f"{path}/train.csv"),
        CSVDataset(f"{path}/test.csv"),
        CSVDataset(f"{path}/eval.csv")
    ]


def _main(path: str, transformer):
    train_df, test_df, eval_df = get_datasets(path)
    train_loader = DataLoader(train_df, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=lambda worker_id: np.random.seed(RANDOM_SEED + worker_id))
    test_loader = DataLoader(test_df, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=lambda worker_id: np.random.seed(RANDOM_SEED + worker_id))
    eval_loader = DataLoader(eval_df, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=lambda worker_id: np.random.seed(RANDOM_SEED + worker_id))

    loss_fn = nn.MSELoss()
    model, train_mse, eval_mse, train_rmse, eval_rmse = train(train_df, train_loader, eval_loader, loss_fn, transformer)

    #evaluate test dataset, unseen data
    test_mse, test_mse_original, test_mse_losses, test_rmse_losses, test_batches = evaluate(test_loader, model, loss_fn, transformer, is_test=True)
    test_rmse = test_mse_original ** 0.5

    return train_mse, eval_mse, train_rmse, eval_rmse, test_mse, test_rmse, test_mse_losses, test_rmse_losses, test_batches

if __name__ == "__main__":
    path = "data/transformed/full_features"
    if IS_WANDB:
        wandb.init(
            project=f"{WANDB_PROJECT_NAME}",
            name="full_norm_exp9_bottleneck",
            config={
                "batch_size": BATCH_SIZE,
                "epoch": EPOCH,
                "lr": LR,
                "loss_fn": "MSELoss",
                "dataset_path": path
            }
        )
    
    target_transformer = joblib.load(f"{path}/house_value_scaler.pkl")

    tr_mse, ev_mse, tr_rmse, ev_rmse, te_mse, te_rmse, te_mse_losses, te_rmse_losses, test_batches = _main(path, target_transformer)
    
    if IS_WANDB:
        for epoch, (mse_t, mse_e, rmse_t, rmse_e) in enumerate(zip(tr_mse, ev_mse, tr_rmse, ev_rmse), 1):
            wandb.log({
                "epoch": epoch,
                "train_MSE": mse_t,
                "eval_MSE": mse_e,
                "train_RMSE": rmse_t,
                "eval_RMSE": rmse_e,
            })


        for batch in range(1, test_batches+1):
            wandb.log({
                "batch": batch,
                "test_MSE": te_mse_losses[batch-1],
                "test_RMSE": te_rmse_losses[batch-1]
            })

        wandb.finish()

    print(f"Test RMSE: {te_rmse:.4f}")

    #plot mse losses
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(tr_mse) + 1), tr_mse, label="Training MSE", linewidth=2)
    plt.plot(range(1, len(ev_mse) + 1), ev_mse, label="Evaluation MSE", linewidth=2)
    plt.title("Training, Evaluation MSE Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    #plot rmse - real losses in €
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(tr_rmse) + 1), tr_rmse, label="Training RMSE", linewidth=2)
    plt.plot(range(1, len(ev_rmse) + 1), ev_rmse, label="Evaluation RMSE", linewidth=2)
    plt.title("Training, Evaluation RMSE Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (€)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    #test plots - mse
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, test_batches + 1), te_mse_losses, label="Test MSE", linewidth=2)
    plt.title("Test MSE Over Batches")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    #test plots - rmse
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, test_batches + 1), te_rmse_losses, label="Test RMSE", linewidth=2)
    plt.title("Test RMSE Over Batches")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (€)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()