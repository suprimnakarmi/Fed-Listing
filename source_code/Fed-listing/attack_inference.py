import glob
import pickle 
import torch 
import torch.nn as nn
import numpy as np
import os
import re
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances


all_paths = glob.glob(os.path.join("logs", "*.pt"))
clients = set()
total_labels = 10
for file in all_paths[:30]:
    clients.add(file.split("_")[1])
clients = list(clients)
print("Clients: ", len(clients))

filtered_paths = []
for cl in clients:
    selected_pth = []
    for path in all_paths:
        # if path.split("_")[0].split("/")[1] == "round1":
        #     continue
        if path.split("_")[1] == cl:
            selected_pth.append(path)
    filtered_paths.append(selected_pth)

# Define NN-based attack model
class AttackNN(nn.Module):
    def __init__(self, R, d, N):
        super().__init__()
        self.fc1 = nn.Linear(R , 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, N)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return F.softmax(self.fc3(x), dim=1)

# Prepare data
X_list = []
for client_wt in filtered_paths:
    sorted_paths = sorted(client_wt, key=lambda x: int(re.search(r'round(\d+)', x).group(1)))
    flat = np.concatenate([
        torch.load(sorted_paths[e], weights_only=False)[-1].ravel()
        for e in range(len(sorted_paths))
    ], axis=0)
    X_list.append(flat)
    print("Flat: ", flat)
    print("Flat shape: ", flat.shape)
    print("\n")

X = torch.tensor(np.stack(X_list), dtype=torch.float32)
# print("Shape of data: ", X.shape)
# print("Shape of data: ", X)

B, R = X.shape
d = R
N = total_labels

def distance_fea(features):
    candidate = X[0]
    eu_dist = torch.norm(features - candidate, dim = 1)
    cos_sim = F.cosine_similarity(features, candidate.unsqueeze(0), dim=1)
    cosine_dist = 1 - cos_sim
    print("Euclidean distance: ", eu_dist)
    print("Cosine distance: ", cosine_dist)

def visualization(features):
    for fea in features:
        plt.hist(fea, bins=10, edgecolor='black')
    plt.show()


# Reshape for NN
X = X.view(B, R)
print("X Shape: ", X.shape)
distance_fea(X)
# visualization(X)


# Initialize and load model
model = AttackNN(R, d, N)
model.load_state_dict(torch.load("attack_model_nn.pth"))
model.eval()

# Inference
with torch.no_grad():
    Y_pred = model(X)

print(Y_pred)

def js_divergence(p,q, eps = 1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p+q)

    kl_pm = F.kl_div(p.log(), m, reduction = "batchmean")
    kl_qm = F.kl_div(q.log(), m, reduction = "batchmean")

    return 0.5 * (kl_pm + kl_qm)

df = pd.read_csv("merged_records.csv")
torch.manual_seed(42)

# shape
num_rows, num_cols = 10, N

# 1) sample uniform(0,1)
ran_tensor= torch.rand(num_rows, num_cols)

# 2) normalize each row so it sums to 1
ran_pred = ran_tensor / ran_tensor.sum(dim=1, keepdim=True)
print("Random tensors: ", ran_pred)

num_class = total_labels
num_clients = 10

for cls in range(num_clients):
    gt = df.iloc[cls*num_class:(cls + 1)*num_class]["Density"]
    gt = torch.tensor(gt.values)
    # print(f"GT for Class {cls}: {gt}")
    pred = Y_pred[cls]
    print(f"GT for Class {cls}: {gt} and \n Prediction: {pred} ")
    print(f"Random Predictions: ", ran_pred[cls])
    ran_p = ran_pred[cls]
    js_metric = js_divergence(gt, pred)
    ran_js_metric = js_divergence(gt,ran_p)
    gt = gt.numpy().reshape(1, -1)
    pred = pred.numpy().reshape(1, -1)
    ran_p = ran_p.numpy().reshape(1, -1)

    
    ran_cos_sim = cosine_similarity(gt,ran_p)
    ran_man_dist = manhattan_distances(gt, ran_p)
    print(gt.shape)
    print(pred.shape)
    
    cos_sim = cosine_similarity(gt, pred)
    man_dist = manhattan_distances(gt, pred)
    print(f"Manhattan distance (with real pred): ", man_dist)
    print(f"Manhattan distance (random): ", ran_man_dist)
    print(f"JS divergence (with real pred): ", js_metric)
    print(f"JS divergence (random): ", ran_js_metric)
    print(f"Cosine similarity (with real pred): ", cos_sim)
    print(f"Cosine similarity (random): ", ran_cos_sim)
    
    print(f"")
    print("\n")