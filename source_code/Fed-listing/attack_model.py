import argparse
import os
import pickle 
import numpy as np
import torch
from pathlib import Path 
import glob
import re
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split 
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)
not_same = []
root_dir ="shadow_logs"
# def flatten_concat(tensor_list):
features, lab_dist = [], []
for folders in os.listdir(root_dir):
    # print(folders)
    if folders[0]!="F":
        continue

    else: 
        all_weights_pth = glob.glob(os.path.join(root_dir,folders,"round*_client*_deltas.pt"))
        all_label_pth = glob.glob(os.path.join(root_dir,folders,"label_dist_client*.pkl"))
        print("Label dist: ", len(all_label_pth))
        # print(all_weights_pth)
        for client_id in range(10):
            selected_paths = []

            for path in all_weights_pth:
                if path.split("_")[2][-1] == str(client_id):
                    selected_paths.append(path)
            
            for lab_pth in all_label_pth:
               if lab_pth.split("_")[-1].split(".")[0][-1] == str(client_id):
                   lab_dist.append(pickle.load(open(lab_pth, "rb")))
                   break

            sorted_paths = sorted(selected_paths, key=lambda x: int(re.search(r'round(\d+)', x).group(1)))
            print("File name: ", folders)
            flat = np.concatenate([torch.load(sorted_paths[e], weights_only=False)[-1].ravel()
                            for e in range(len(sorted_paths))
                            ]) 
            features.append(flat)
            # if flat.shape[0] != 4800:
            #     not_same.append(folders)
            # break
            # print("File name: ", folders)
            print("Flat: ", flat)
            print("Flat shape: ", flat.shape)
            print("\n")

# print("Not same: ", not_same)
X = torch.tensor(np.stack(features), dtype=torch.float32)  # [B, R, C, F]
Y = torch.tensor(np.stack(lab_dist), dtype=torch.float32)

pd.DataFrame(Y.numpy(), columns=[f"class_{i}" for i in range(Y.shape[1])]).to_csv("shadow_data_dist.csv", index = False)

print("X shape: ", X.shape)
print("Y shape: ", Y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
B, R= X.shape
input_dim = R 
X = X.view(B, input_dim)
N = Y.shape[1]

class AttackNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return F.softmax(self.fc3(x), dim=1)


model = AttackNN(input_dim=input_dim, hidden_dim=256, output_dim=N)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def js_divergence(p,q, eps = 1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p+q)

    kl_pm = F.kl_div(p.log(), m, reduction = "batchmean")
    kl_qm = F.kl_div(q.log(), m, reduction = "batchmean")

    return 0.5 * (kl_pm + kl_qm)


def loss_fn(y_pred, y_true, a, b, c):
    mean_loss = torch.norm(y_pred.mean(0) - y_true.mean(0), p=1)
    mse_mean = F.mse_loss(y_pred.mean(0), y_true.mean(0))
    js = js_divergence(y_pred, y_true)
    return a * mse_mean + b * js + c * mean_loss

    # return mean_loss + 3 * js + 1 * var_loss
values = [0.5, 1, 1.5, 2]
least_err = 1000
best_values = {}
# Training loop
for a in values:
    for b in values:
        for c in values:
            print(f"Running experiment for a --> {a}, b --> {b}, and c --> {c}")
            for epoch in range(1, 51):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = loss_fn(y_pred, y_train,a,b,c)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch:2d} — Loss: {loss.item():.4f}")

                if epoch%5==0:
                    model.eval()
                    with torch.no_grad():
                        Y_hat = model(X_test)
                        t_loss = loss_fn(Y_hat, y_test,a,b,c)
                        print(f"Total loss: {t_loss} ")

            print("\n")
            if t_loss < least_err:
                least_err = t_loss
                print(f"Least loss obtained at  a --> {a}, b --> {b}, and c --> {c}")
                best_values['a'] = a
                best_values['b'] = b 
                best_values['c'] = c 
                best_values['loss'] = t_loss
                torch.save(model.state_dict(), "attack_model_nn.pth")
                print("Saved attack_model_nn.pth")

print(f"Best values obtained at: {best_values}")


