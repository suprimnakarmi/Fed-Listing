import argparse, torch, torch.nn.functional as F, numpy as np
import flwr as fl
from flwr.client import NumPyClient
from torch_geometric.nn    import GCNConv
from torch_geometric.data  import Data
from torch_geometric.utils import subgraph
import os
import pickle

AUX_PATH          = "auxiliary_data.pt"            # full sub-graph
NUM_CLIENTS       = 10
LOCAL_EPOCHS      = 30
LR                = 0.01
HIDDEN_DIM        = 16
DEVICE            = torch.device("mps")

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hid, num_cls):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hid)
        self.conv2 = GCNConv(hid, num_cls)

    def forward(self, x, e):
        x = self.conv1(x, e).relu()
        x = F.dropout(x, 0.5, self.training)
        return self.conv2(x, e)
    
class AuxClient(NumPyClient):
    def __init__(self, cid: int, data: Data, train_idx: torch.Tensor):
        self.cid   = cid
        self.data  = data.to(DEVICE)
        self.train = train_idx.to(DEVICE)
        self.model = GCN(data.num_node_features, HIDDEN_DIM,
                         int(data.y.max().item()) + 1).to(DEVICE)
        self.opt   = torch.optim.Adam(self.model.parameters(), lr=LR)

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, params, config=None):
        sd = {k: torch.as_tensor(p) for k, p
              in zip(self.model.state_dict().keys(), params)}
        self.model.load_state_dict(sd, strict=True)

    def fit(self, params, _):
        self.set_parameters(params)
        self.model.train()
        for _ in range(LOCAL_EPOCHS):
            self.opt.zero_grad()
            out  = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out[self.train], self.data.y[self.train])
            loss.backward(); self.opt.step()
        return self.get_parameters(), len(self.train), {}

    def evaluate(self, params, _):
        self.set_parameters(params)
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            logits, labels = out[self.train], self.data.y[self.train]
            loss = F.cross_entropy(logits, labels).item()
            acc  = (logits.argmax(1) == labels).float().mean().item()
        print(f"[Client {self.cid}]  loss={loss:.4f}, acc={acc:.4f}")
        return loss, len(self.train), {"accuracy": acc}


def build_client_dataset(cid: int):
    aux = torch.load(AUX_PATH, weights_only=False)

    # Deterministic global permutation once
    gperm = torch.load("aux_permutation.pt") if \
            os.path.exists("aux_permutation.pt") else None
    if gperm is None:
        gperm = torch.randperm(aux.num_nodes)
        torch.save(gperm, "aux_permutation.pt")

    chunk_size = int(np.ceil(aux.num_nodes / NUM_CLIENTS))
    chunk = gperm[cid*chunk_size:(cid+1)*chunk_size]
    # build induced sub-graph
    eidx, _ = subgraph(chunk, aux.edge_index, relabel_nodes=True,
                       num_nodes=aux.num_nodes)
    d = Data(x=aux.x[chunk],
             y=aux.y[chunk],
             edge_index=eidx)
    num_classes = int(aux.y.max().item()) + 1
    counts      = torch.bincount(d.y, minlength=num_classes).cpu().numpy()
    distribution = counts / counts.sum()           # NumPy 1-D array

    out_dir = "shadow_logs/FL12"
    os.makedirs(out_dir, exist_ok=True)           # ← ensure the directory exists
    out_path = os.path.join(out_dir, f"label_dist_client{cid}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(distribution, f)

    print(f"[Client {cid}] saved label distribution → {out_path}") 

    train_idx = torch.arange(d.num_nodes)          # full sub-graph is "train"
    return d, train_idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True, help="Client ID (0-9)")
    args = ap.parse_args()

    data, idx = build_client_dataset(args.cid)
    client = AuxClient(args.cid, data, idx)

    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
