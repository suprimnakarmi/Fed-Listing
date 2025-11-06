import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances


# Getting global weights 
model = "sage"
num_class=7
df = pd.read_csv("global_all_layer_weights_new")

# Last layer weight of all the clients in each round 
df_client = pd.read_excel("last_layer_weights_new.xlsx", engine="openpyxl")

# Class wise 
df_class = pd.read_csv("classwise_last_layer_weights_new.csv")

# Combined auxiliary dataset weights
df_allaux = pd.read_csv("all_aux_last_layer_weights_new.csv")
df_gt_dist = pd.read_csv("merged_records.csv")
print(df)
# Filter for round 19 and last layer's weight
if model == "sage":
    last_layer_key = "conv2.lin_l.weight"
elif model == "gin":
    last_layer_key = "conv2.nn.2.weight"
else:
    last_layer_key = "conv2.lin.weight"
round9_last_layer = df[
    (df["round"] == 19) &
    (df["layer"] == last_layer_key) &
    (df["type"] == "weight")
]
print("here",round9_last_layer)

# Extract weight tensor
w_cols = [col for col in round9_last_layer.columns if col.startswith("w_")]
last_layer_weights = torch.tensor(round9_last_layer[w_cols].values[0], dtype=torch.float)

print(f"Loaded last layer weights for '{last_layer_key}' at round 9 — shape: {last_layer_weights.shape}")
secondlast_global_wt = last_layer_weights[~torch.isnan(last_layer_weights)]

# round_10 = df_client[df_client["round"] == 20]
secondlast_global_wt = secondlast_global_wt.numpy()
print("DF client", df_client)
all_client_end = df_client[df_client["round"]==20]
# print(all_client_end)
wt_cols = [col for col in all_client_end.columns if col.startswith("w_")]
# print(wt_cols)
all_cl_last_wt = torch.tensor(all_client_end[wt_cols].values, dtype=torch.float)
# print("all clients wt", all_cl_last_wt)

df_weight = df_allaux[df_allaux["type"] == "weight2"]
# print("WTTTTT: ", df_weight)
weight_columns = [col for col in df_weight.columns if col.startswith("w_")]
weight_vector = df_weight.iloc[0][weight_columns].dropna().astype(float).values


gradient_base_allaux = (secondlast_global_wt - weight_vector)/0.01
# print("gradient base: ", gradient_base_allaux)

# print("All aux weight: ", weight_vector)
# print("Last layer all clients: ", round_10)
df_class_weights = df_class[df_class['type'] == 'weight2']
class_0_wt = df_class_weights.iloc[0][[col for col in df_class_weights.columns if col.startswith("w_")]]
# Start the figure
plt.figure(figsize=(10, 6))

all_clients_dist = []
for i in range(len(all_cl_last_wt)):
    client_wt = all_cl_last_wt[i]
    # weights_only = client[[col for col in client.index if col.startswith("w_")]]
    # weight_tensor = torch.tensor(weights_only.values, dtype=torch.float))
    # print(client_wt)
    g_total = (client_wt - secondlast_global_wt)/0.01

    gradient_bases_each_class = []
    df_class_weights = df_class[df_class['type'] == 'weight2']
    # print("All class: ", df_class_weights)
    # print("Client index: ", client.index)

    for j in range(len(df_class_weights)):
        # print(df_class_weights.iloc[i])
        cols = [c for c in df_class_weights.columns if c.startswith("w_")]
        class_weights = df_class_weights.iloc[j][cols]
    
        # Drop any NaNs, cast to float, then grab the numpy array
        class_weights = class_weights.dropna().astype(float).values
    
        # print("Class weights:", class_weights)

        # print("\n Class weights: ",class_weights)
        gradient_bases_each_class.append((secondlast_global_wt - class_weights)/0.01)
    
    # print("gradient bases", gradient_bases_each_class)
    def remaining_class_decomposition(gtarget, gradient_bases_gc_list, gu, epochs=1000, lr=0.0001):
        num_classes = len(gradient_bases_gc_list)
        eta_c = np.random.rand(num_classes)
        # print("Eta_c: ", eta_c)
        eta_u = np.random.rand()
        for a in range(epochs):
            # Compute estimated gradient
            g_est = sum(eta_c[i] * gradient_bases_gc_list[i] for i in range(num_classes)) + eta_u * gu
            # Compute difference
            # print("G_est shape",g_est.shape)
            # print("gtarget shape",gtarget.shape)
            gtarget = np.array(gtarget)
            diff = g_est - gtarget
            # print("Diff: ", diff)

            # Update eta_c for each class
            for i in range(num_classes):
                # print("Gradient bases: ", gradient_bases_gc_list[i] )
                grad = 2 * np.dot(diff, gradient_bases_gc_list[i])
                eta_c[i] -= lr * grad
                eta_c[i] = max(0, eta_c[i])  # keep non-negative

            # Update eta_u
            grad_u = 2 * np.dot(diff, gu)
            eta_u -= lr * grad_u
            eta_u = max(0, eta_u)
        total = np.sum(eta_c) 
        print("eta u", eta_u)
        print("eta_c",eta_c)
        proportions = {f"class_{i}": (eta_c[i] + eta_u) / (total + num_classes * eta_u) for i in range(num_classes)}
        print("\n")
        return proportions

    label_distribution = remaining_class_decomposition(g_total, gradient_bases_each_class, gradient_base_allaux)
    values = [label_distribution[f"class_{i}"] for i in range(num_class)]
    all_clients_dist.append(values)
    print(f"Client: {i+1} \t label_distribution:{label_distribution}")
    print(f"Sum of proportions: {sum(label_distribution.values())}\n")

def js_divergence(p,q, eps = 1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p+q)

    kl_pm = F.kl_div(p.log(), m, reduction = "batchmean")
    kl_qm = F.kl_div(q.log(), m, reduction = "batchmean")

    return 0.5 * (kl_pm + kl_qm)

clients=10
all_gt= []

# print(len(all_clients_dist))
for cls in range(clients):
    gt = df_gt_dist.iloc[cls*num_class:(cls + 1) * num_class]["Density"]
    gt = torch.tensor(gt.values)
    all_gt.append(gt)
    pred = torch.tensor(all_clients_dist[cls])
    print(f"GT for Client {cls+1}: {gt} and \n Prediction: {pred} ")
    # print(f"Random Predictions: ", ran_pred[cls])
    js_metric = js_divergence(gt, pred)
    # ran_js_metric = js_divergence(gt,ran_p)
    gt = gt.numpy().reshape(1, -1)
    pred = pred.numpy().reshape(1, -1)
    # ran_p = ran_p.numpy().reshape(1, -1)

    
    # ran_cos_sim = cosine_similarity(gt,ran_p)
    # ran_man_dist = manhattan_distances(gt, ran_p)
    print(gt.shape)
    print(pred.shape)
    
    # print(f"JS divergence (random): ", ran_js_metric)
    cos_sim = cosine_similarity(gt, pred)
    man_dist = manhattan_distances(gt, pred)
    print(f"Manhattan distance (with real pred): ", man_dist)
    print(f"JS divergence (with real pred): ", js_metric)
    print(f"Cosine similarity (with real pred): ", cos_sim)
    
    # print(f"")
    print("\n")
# print(round_10)
# print(all_gt)