import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Amazon
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch.nn import Linear, Sequential, ReLU
import flwr as fl
from flwr.client import NumPyClient
import pandas as pd
from torch_geometric.datasets import Planetoid

def stratified_aux_indices(labels, reserve_frac, seed=42):
    N = labels.size(0)
    N_aux = int(round(N * reserve_frac))
    class_idxs = [
        (labels == c).nonzero(as_tuple=False).view(-1)
        for c in range(int(labels.max())+1)
    ]
    rng = torch.Generator().manual_seed(seed)
    parts = []; rem = N_aux
    for i, idx in enumerate(class_idxs):
        # desired per-class count
        cnt = int(round(len(idx)/N * N_aux)) if i < len(class_idxs)-1 else rem
        rem -= cnt
        perm = idx[torch.randperm(len(idx), generator=rng)]
        parts.append(perm[:cnt])
    aux = torch.cat(parts)
    return aux[torch.randperm(len(aux), generator=rng)]

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, num_classes)
    def forward(self, x, edge_index, verbose=False):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

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

class Client(NumPyClient):
    def __init__(self, model, data, train_idx, client_id):
        self.model = model
        self.data = data
        self.train_idx = train_idx
        self.client_id = client_id
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config=None):
        return [p.cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        sd = {k: torch.tensor(v) for k,v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(sd, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(30):
            self.opt.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out[self.train_idx], self.data.y[self.train_idx])
            loss.backward(); self.opt.step()
        return self.get_parameters(), len(self.train_idx), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            mask = self.data.test_mask
            loss = F.cross_entropy(out[mask], self.data.y[mask]).item()
            pred = out[mask].argmax(1)
            acc = (pred==self.data.y[mask]).sum().item()/mask.sum().item()
        return float(loss), int(mask.sum().item()), {"accuracy":acc}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--reserve-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.1)
    args = p.parse_args()

    # Uncomment each data for experimentation
    # dataset = Amazon(root="data/Amazon", name="Computers")
    dataset = Planetoid(root="data/Cora", name="Cora")
    # dataset = Planetoid(root='data/PubMed', name='PubMed')
    # dataset = Planetoid(root='data/CiteSeer', name='CiteSeer')
    data = dataset[0]
    torch.manual_seed(42) 
    np.random.seed(42)

   # Split the data to create Auxiliary dataset
    aux_idx = stratified_aux_indices(data.y, args.reserve_fraction)
    aux_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    aux_mask[aux_idx] = True

    # Create train/test pool from the graph dataset
    pool = (~aux_mask).nonzero(as_tuple=False).view(-1)
    n_test = int(len(pool)*args.test_fraction)
    test_pool = pool[:n_test]; train_pool = pool[n_test:]

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_pool] = True
    data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[test_pool] = True

    # Split data into different proportion
    y_pool = data.y[train_pool]
    num_cls = dataset.num_classes
    cls2idx = {c: train_pool[(y_pool==c).nonzero(as_tuple=False).view(-1)]
               for c in range(num_cls)}

    cid = args.client_id
    if cid==0:
        # equal proportion
        m = min(len(v) for v in cls2idx.values())
        local = torch.cat([v[:m] for v in cls2idx.values()])
    elif cid in {1,2,3,5,6}:
        # random proportion
        perm = train_pool[torch.randperm(len(train_pool))]
        local = torch.chunk(perm, args.num_clients)[cid]
    elif cid==4:
        # missing class 3
        candidates = torch.cat([cls2idx[c] for c in range(num_cls) if c!=3])
        perm = candidates[torch.randperm(len(candidates))]
        local = perm[:len(train_pool)//args.num_clients]
    elif cid==7:
        # only class 7
        local = cls2idx.get(7, torch.tensor([],dtype=torch.long))
    else:
        # dominant class split
        counts = torch.bincount(y_pool, minlength=num_cls)
        dom = int(counts.argmax())
        total = len(train_pool)//args.num_clients
        k_dom = int(0.8*total); k_rem = total-k_dom
        dom_idx = cls2idx[dom][torch.randperm(len(cls2idx[dom]))[:k_dom]]
        others = torch.cat([cls2idx[c] for c in range(num_cls) if c!=dom])
        rem   = others[torch.randperm(len(others))[:k_rem]]
        local = torch.cat([dom_idx, rem])[torch.randperm(total)]

    # Saving the auxiliary dataset
    if cid==0:
        e2, _ = subgraph(aux_idx, data.edge_index, relabel_nodes=True)
        torch.save(Data(x=data.x[aux_idx], y=data.y[aux_idx], edge_index=e2),
                   "auxiliary_data.pt")

    # Store the distribution information
    recs=[]
    arr = data.y[local].numpy()
    u,c = np.unique(arr,return_counts=True)
    for cls,count in zip(u,c):
        recs.append({"client_id":cid,"Class":int(cls),
                     "Frequency":int(count),
                     "Density":count/len(arr)})
        
    # fill missing classes with frequency and density 0, required for performance calculation
    for cls in range(num_cls):
        if cls not in u:
            recs.append({"client_id":cid,"Class":cls,
                         "Frequency":0,"Density":0.0})
    pd.DataFrame(recs).to_csv(f"records_{cid}.csv",index=False)

    # Model/ FL training initialization
    model = GraphSAGE(dataset.num_node_features, 16, num_cls)
    client = Client(model, data, local, cid)
    fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=client)


if __name__=="__main__":
    main()
