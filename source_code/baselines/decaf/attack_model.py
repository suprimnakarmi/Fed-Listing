import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.utils import to_networkx
from torch.nn import Linear, Sequential, ReLU
import networkx as nx
import numpy as np

# Load the auxiliary dataset
model_name = "sage"
aux_data = torch.load("auxiliary_data.pt", weights_only=False)
df = pd.read_csv("global_all_layer_weights_new")
print(df)
if model_name == "sage":
    weight_row1 = df[(df["round"] == 19) & (df["layer"] == "conv1.lin_r.weight")].iloc[0]
    bias_row1 = df[(df["round"] == 19) & (df["layer"] == "conv1.lin_l.bias")].iloc[0]
    weight_row2 = df[(df["round"] == 19) & (df["layer"] == "conv2.lin_r.weight")].iloc[0]
    bias_row2 = df[(df["round"] == 19) & (df["layer"] == "conv2.lin_l.bias")].iloc[0]
elif model_name =="gin":
    weight_row1 = df[(df["round"] == 19) & (df["layer"] == "conv1.nn.0.weight")].iloc[0]
    bias_row1 = df[(df["round"] == 19) & (df["layer"] == "conv1.nn.0.bias")].iloc[0]
    weight_row2 = df[(df["round"] == 19) & (df["layer"] == "conv2.nn.2.weight")].iloc[0]
    bias_row2 = df[(df["round"] == 19) & (df["layer"] == "conv2.nn.2.bias")].iloc[0]
else: 
    weight_row1 = df[(df["round"] == 19) & (df["layer"] == "conv1.lin.weight")].iloc[0]
    bias_row1 = df[(df["round"] == 19) & (df["layer"] == "conv1.bias")].iloc[0]
    weight_row2 = df[(df["round"] == 19) & (df["layer"] == "conv2.lin.weight")].iloc[0]
    bias_row2 = df[(df["round"] == 19) & (df["layer"] == "conv2.bias")].iloc[0]
    
print(weight_row1)
# NA exists as weight of initial layers has higer dimensions.
weight1 = weight_row1.filter(like="w_").dropna().values.astype(np.float32)
bias1 = bias_row1.filter(like="w_").dropna().values.astype(np.float32)
weight2 = weight_row2.filter(like="w_").dropna().values.astype(np.float32)
bias2 = bias_row2.filter(like="w_").dropna().values.astype(np.float32)

labels = aux_data.y
unique_classes, counts = torch.unique(labels, return_counts=True)

print("\nClass distribution in auxiliary dataset:")
for cls, count in zip(unique_classes.tolist(), counts.tolist()):
    print(f"Class {cls}: {count} samples")


# GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats)
        self.conv2 = SAGEConv(hidden_feats, num_classes)

    def forward(self, x, edge_index, verbose=False):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GIN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super().__init__()
        nn1 = Sequential(Linear(in_feats, hidden_feats), ReLU(), Linear(hidden_feats, hidden_feats))
        nn2 = Sequential(Linear(hidden_feats, hidden_feats), ReLU(), Linear(hidden_feats, num_classes))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index, verbose=False):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

in_feats = aux_data.num_node_features
num_classes = int(aux_data.y.max().item()) + 1 
# model = GCN(in_feats= in_feats, hidden_feats=16, num_classes=num_classes)
model = GraphSAGE(in_feats= in_feats, hidden_feats=16, num_classes=num_classes)
state_dict = model.state_dict()

# Initialize model, loss, optimizer
in_feats = aux_data.x.shape[1]
hidden_feats = 16
num_classes = aux_data.y.max().item() + 1

# For saving last layer weights
records = []
NUM_EPOCHS = 1 # Epoch 1 used originally in the paper

# Train on each class-specific subset
for cls in range(num_classes):
    print(f"\n Training on class {cls}")
    if model_name == "sage":
        state_dict['conv1.lin_r.weight'] = torch.tensor(weight1).reshape(state_dict['conv1.lin_r.weight'].shape)
        state_dict['conv1.lin_l.bias'] = torch.tensor(bias1).reshape(state_dict['conv1.lin_l.bias'].shape)

        state_dict['conv2.lin_r.weight'] = torch.tensor(weight2).reshape(state_dict['conv2.lin_r.weight'].shape)
        state_dict['conv2.lin_l.bias'] = torch.tensor(bias2).reshape(state_dict['conv2.lin_l.bias'].shape)

    elif model_name =="gin":
        state_dict['conv1.nn.0.weight'] = torch.tensor(weight1).reshape(state_dict['conv1.nn.0.weight'].shape)
        state_dict['conv1.nn.0.bias'] = torch.tensor(bias1).reshape(state_dict['conv1.nn.0.bias'].shape)

        state_dict['conv2.nn.2.weight'] = torch.tensor(weight2).reshape(state_dict['conv2.nn.2.weight'].shape)
        state_dict['conv2.nn.2.bias'] = torch.tensor(bias2).reshape(state_dict['conv2.nn.2.bias'].shape)

    else: 
        print(f"Keys {state_dict.keys()}")
        state_dict['conv1.lin.weight'] = torch.tensor(weight1).reshape(state_dict['conv1.lin.weight'].shape)
        state_dict['conv1.bias'] = torch.tensor(bias1).reshape(state_dict['conv1.bias'].shape)

        state_dict['conv2.lin.weight'] = torch.tensor(weight2).reshape(state_dict['conv2.lin.weight'].shape)
        state_dict['conv2.bias'] = torch.tensor(bias2).reshape(state_dict['conv2.bias'].shape)
    model.load_state_dict(state_dict)

    # Get indices for current class
    class_mask = (aux_data.y == cls)
    class_idx = class_mask.nonzero(as_tuple=False).view(-1)

    # Create a model for this class
    # model = GCN(in_feats=in_feats, hidden_feats=hidden_feats, num_classes=num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(1, NUM_EPOCHS+1):
        optimizer.zero_grad()
        out  = model(aux_data.x, aux_data.edge_index)
        loss = F.cross_entropy(out[class_idx], aux_data.y[class_idx])
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

    # Save last layer weights and biases
    state_dict = model.state_dict()
    if model_name =="sage":
        conv1_w = state_dict["conv1.lin_r.weight"].flatten().detach().cpu().numpy()
        conv1_b = state_dict["conv1.lin_l.bias"].flatten().detach().cpu().numpy()
        conv2_w = state_dict["conv2.lin_r.weight"].flatten().detach().cpu().numpy()
        conv2_b = state_dict["conv2.lin_l.bias"].flatten().detach().cpu().numpy()
    elif model_name =="gin":
        conv1_w = state_dict["conv1.nn.0.weight"].flatten().detach().cpu().numpy()
        conv1_b = state_dict["conv1.nn.0.bias"].flatten().detach().cpu().numpy()
        conv2_w = state_dict["conv2.nn.2.weight"].flatten().detach().cpu().numpy()
        conv2_b = state_dict["conv2.nn.2.bias"].flatten().detach().cpu().numpy()
    else:
        conv1_w = state_dict["conv1.lin.weight"].flatten().detach().cpu().numpy()
        conv1_b = state_dict["conv1.bias"].flatten().detach().cpu().numpy()
        conv2_w = state_dict["conv2.lin.weight"].flatten().detach().cpu().numpy()
        conv2_b = state_dict["conv2.bias"].flatten().detach().cpu().numpy() 

    weight_row1 = {
        "class": cls,
        "type": "weight1",
        **{f"w_{i}": v for i, v in enumerate(conv1_w)}
    }

    bias_row1 = {
        "class": cls,
        "type": "bias1",
        **{f"w_{i}": v for i, v in enumerate(conv1_b)}
    }

    weight_row2 = {
        "class": cls,
        "type": "weight2",
        **{f"w_{i}": v for i, v in enumerate(conv2_w)}
    }

    bias_row2 = {
        "class": cls,
        "type": "bias2",
        **{f"w_{i}": v for i, v in enumerate(conv2_b)}
    }

    records.append(weight_row1)
    records.append(bias_row1)
    records.append(weight_row2)
    records.append(bias_row2)

# Save to CSV
df = pd.DataFrame(records)
df.to_csv("classwise_last_layer_weights_new.csv", index=False)
print("\n Saved weights to 'classwise_last_layer_weights.csv'")

# Visualization of the graph
print("\n Visualizing auxiliary dataset...")
G = to_networkx(aux_data, to_undirected=True)
y = aux_data.y.detach().cpu().numpy()


# Gradient Base: Train on all auxiliary data 
print("\n Training gradient base model on full auxiliary dataset...")
record_allaux = []

# Initialize model
# base_model = GCN(in_feats=in_feats, hidden_feats=hidden_feats, num_classes=num_classes)
base_model = GraphSAGE(in_feats=in_feats, hidden_feats=hidden_feats, num_classes=num_classes)

optimizer = torch.optim.SGD(base_model.parameters(), lr=0.01)
base_model.train()
NUM_EPOCHS_all = 1
# Train for 1 epoch on all auxiliary nodes
for epoch in range(1, NUM_EPOCHS_all+1):
    optimizer.zero_grad()
    out = base_model(aux_data.x, aux_data.edge_index)
    loss = F.cross_entropy(out, aux_data.y)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

# Save final gradient base weights
base_state = base_model.state_dict()
if model_name == "sage":
    conv1_w = state_dict["conv1.lin_r.weight"].flatten().detach().cpu().numpy()
    conv1_b = state_dict["conv1.lin_l.bias"].flatten().detach().cpu().numpy()
    conv2_w = state_dict["conv2.lin_r.weight"].flatten().detach().cpu().numpy()
    conv2_b = state_dict["conv2.lin_l.bias"].flatten().detach().cpu().numpy()
elif model_name == "gin":
    conv1_w = state_dict["conv1.nn.0.weight"].flatten().detach().cpu().numpy()
    conv1_b = state_dict["conv1.nn.0.bias"].flatten().detach().cpu().numpy()
    conv2_w = state_dict["conv2.nn.2.weight"].flatten().detach().cpu().numpy()
    conv2_b = state_dict["conv2.nn.2.bias"].flatten().detach().cpu().numpy()
else:
    conv1_w = state_dict["conv1.lin.weight"].flatten().detach().cpu().numpy()
    conv1_b = state_dict["conv1.bias"].flatten().detach().cpu().numpy()
    conv2_w = state_dict["conv2.lin.weight"].flatten().detach().cpu().numpy()
    conv2_b = state_dict["conv2.bias"].flatten().detach().cpu().numpy()

# Append to CSV records with special tag
weight_row1 = {
        "class": cls,
        "type": "weight1",
        **{f"w_{i}": v for i, v in enumerate(conv1_w)}
    }

bias_row1 = {
        "class": cls,
        "type": "bias1",
        **{f"w_{i}": v for i, v in enumerate(conv1_b)}
    }

weight_row2 = {
        "class": cls,
        "type": "weight2",
        **{f"w_{i}": v for i, v in enumerate(conv2_w)}
    }

bias_row2 = {
        "class": cls,
        "type": "bias2",
        **{f"w_{i}": v for i, v in enumerate(conv2_b)}
    }
record_allaux.append(weight_row1)
record_allaux.append(bias_row1)
record_allaux.append(weight_row2)
record_allaux.append(bias_row2)

df = pd.DataFrame(record_allaux)
df.to_csv("all_aux_last_layer_weights_new.csv", index=False)
