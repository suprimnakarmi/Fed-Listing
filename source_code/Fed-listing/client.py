import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
from torch.nn import Linear, Sequential, ReLU
import flwr as fl
from flwr.client import NumPyClient
import pandas as pd

# GNN model
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, num_classes)

    def forward(self, x, edge_index, verbose=False):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats)
        self.conv2 = SAGEConv(hidden_feats, num_classes)

    def forward(self, x, edge_index, verbose=False):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

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
        return self.conv2(x, edge_index)

class GAT(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, heads=2):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, heads=heads)
        self.conv2 = GATConv(hidden_feats * heads, num_classes, heads=1)

    def forward(self, x, edge_index, verbose=False):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

# Flower client initialization
class Client(NumPyClient):
    def __init__(self, model, data, train_idx, client_id):
        self.model = model
        self.data = data
        self.train_idx = train_idx
        self.client_id = client_id
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        for i in range(30):
            self.optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out[self.train_idx], self.data.y[self.train_idx])
            loss.backward()
            self.optimizer.step()
            if i % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(self.data.x, self.data.edge_index)[self.train_idx]
                    labels = self.data.y[self.train_idx]
                    preds = logits.argmax(dim=1)
                    acc = (preds == labels).sum().item() / len(self.train_idx)
                    print(f"[Client {self.client_id}] Train Loss: {loss.item():.4f}, Train Acc: {acc:.4f}")
        return self.get_parameters(), len(self.train_idx), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            mask = self.data.test_mask
            logits = out[mask]
            labels = self.data.y[mask]
            loss = F.cross_entropy(logits, labels).item()
            preds = logits.argmax(dim=1)
            acc = (preds == labels).sum().item() / mask.sum().item()
            print(f"[Client {self.client_id}] Test Loss: {loss:.4f}, Test Acc: {acc:.4f}")
        return float(loss), int(mask.sum().item()), {"accuracy": float(acc)}

# Separating the auxiliary dataset
def stratified_aux_indices(labels, reserve_frac, seed=42):
    N_total      = labels.size(0)
    N_aux_target = int(round(N_total * reserve_frac))
    class_indices = [(labels == c).nonzero(as_tuple=False).view(-1)
                     for c in range(labels.max().item() + 1)]
    rng = torch.Generator().manual_seed(seed)
    aux_chunks, remaining = [], N_aux_target
    for c, idx in enumerate(class_indices):
        p_c = len(idx) / N_total
        k_c = int(round(p_c * N_aux_target))
        if c == len(class_indices) - 1:
            k_c = remaining
        remaining -= k_c
        perm = idx[torch.randperm(len(idx), generator=rng)]
        aux_chunks.append(perm[:k_c])
    aux_idx = torch.cat(aux_chunks)
    return aux_idx[torch.randperm(len(aux_idx), generator=rng)]


def main():
    parser = argparse.ArgumentParser(description="Flower client with heterogeneity")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID (0-9)")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--reserve-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--GNN-architecture", type=str, default="GCN",
                        help="One of: GCN, SAGE, GIN, GAT")
    args = parser.parse_args()

    # Load dataset (example: PubMed; change root for Cora/CiteSeer)
    dataset = Planetoid(root='data/PubMed', name='PubMed')
    data    = dataset[0]
    num_total = data.num_nodes

    torch.manual_seed(42)
    np.random.seed(42)

    # Create auxiliary indices
    aux_idx = stratified_aux_indices(data.y, reserve_frac=args.reserve_fraction)
    aux_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    aux_mask[aux_idx] = True
    train_test_pool = (~aux_mask).nonzero(as_tuple=False).view(-1)

    # Split into train/test
    num_test = int(len(train_test_pool) * args.test_fraction)
    test_pool = train_test_pool[:num_test]
    train_pool = train_test_pool[num_test:]

    data.train_mask = torch.zeros(num_total, dtype=torch.bool)
    data.train_mask[train_pool] = True
    data.test_mask  = torch.zeros(num_total, dtype=torch.bool)
    data.test_mask[test_pool]  = True

    # Save auxiliary data for client 0
    if args.client_id == 0:
        aux_edge, _ = subgraph(aux_idx, data.edge_index, relabel_nodes=True)
        torch.save(Data(x=data.x[aux_idx], y=data.y[aux_idx], edge_index=aux_edge),
                   "auxiliary_data.pt")
        dist = torch.bincount(data.y[aux_idx], minlength=dataset.num_classes).float()
        torch.save(dist / dist.sum(), "aux_label_dist.pt")
        print(f"[Client 0] Saved auxiliary dataset with {len(aux_idx)} nodes.")

    y_train_pool = data.y[train_pool]
    num_classes  = dataset.num_classes
    class_indices = {c: train_pool[y_train_pool == c] for c in range(num_classes)}

    # Heterogeneous client partitions
    if args.client_id == 0:
        min_count = min(len(idxs) for idxs in class_indices.values())
        local_train_idx = torch.cat([idxs[:min_count] for idxs in class_indices.values()])
        # Clients with random distribution
    elif args.client_id in {1,2,3,5,6}:
        permuted = train_pool[torch.randperm(len(train_pool))]
        splits   = torch.chunk(permuted, args.num_clients)
        local_train_idx = splits[args.client_id]
        # Client with one missing class
    elif args.client_id == 4:
        present_idxs = torch.cat([class_indices[c] for c in range(num_classes) if c != 3])
        perm = present_idxs[torch.randperm(len(present_idxs))]
        target_size = len(train_pool) // args.num_clients
        local_train_idx = perm[:target_size]
        # Client with only one class
    elif args.client_id == 7: 
        local_train_idx = class_indices[2]
    else:  # Clients 8 and 9
        counts = torch.bincount(y_train_pool, minlength=num_classes)
        dom = int(counts.argmax())
        total = len(train_pool) // args.num_clients
        n_dom = int(0.8 * total)
        dom_idx = class_indices[dom][:n_dom]
        other_idxs = torch.cat([class_indices[c] for c in range(num_classes) if c != dom])
        rem_idx = other_idxs[torch.randperm(len(other_idxs))][:total-n_dom]
        local_train_idx = torch.cat([dom_idx, rem_idx])
        local_train_idx = local_train_idx[torch.randperm(len(local_train_idx))]

    # Log label distribution
    records = []
    label_numpy = data.y[local_train_idx].cpu().numpy()
    uni_ele, cnts = np.unique(label_numpy, return_counts=True)
    all_cls = 0
    for elem, count in zip(uni_ele, cnts):
        while elem > all_cls:
            records.append({"client_id": args.client_id,
                            "Class": all_cls,
                            "Frequency": 0,
                            "Density": 0.0})
            all_cls += 1
        records.append({"client_id": args.client_id,
                        "Class": elem,
                        "Frequency": int(count),
                        "Density": float(count / cnts.sum())})
        all_cls += 1
    while all_cls < dataset.num_classes:
        records.append({"client_id": args.client_id,
                        "Class": all_cls,
                        "Frequency": 0,
                        "Density": 0.0})
        all_cls += 1
    pd.DataFrame(records).to_csv(f"records_{args.client_id}.csv", index=False)

    # Build model
    arch = args.GNN_architecture.upper()
    if arch == "GCN":
        model = GCN(dataset.num_node_features, 16, dataset.num_classes)
    elif arch == "SAGE":
        model = GraphSAGE(dataset.num_node_features, 16, dataset.num_classes)
    elif arch == "GIN":
        model = GIN(dataset.num_node_features, 16, dataset.num_classes)
    elif arch == "GAT":
        model = GAT(dataset.num_node_features, 16, dataset.num_classes)
    else:
        raise ValueError(f"Unknown GNN architecture: {args.GNN_architecture}")

    # Start Flower client
    client = Client(model, data, local_train_idx, client_id=args.client_id)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
