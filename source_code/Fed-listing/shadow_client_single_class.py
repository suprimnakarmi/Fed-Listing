import argparse, os, pickle, numpy as np, torch, torch.nn.functional as F
import flwr as fl
from flwr.client import NumPyClient
from torch_geometric.nn    import GCNConv
from torch_geometric.data  import Data
from torch_geometric.utils import subgraph
import math
import random

AUX_PATH          = "auxiliary_data.pt"
NUM_CLIENTS       = 10
LOCAL_EPOCHS      = 30
LR                = 0.01
HIDDEN_DIM        = 16
DEVICE            = torch.device("mps")          


GLOBAL_NUM_CLASSES = int(torch.load(AUX_PATH, weights_only=False).y.max().item()) + 1 # Use global class size when needed

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
                         GLOBAL_NUM_CLASSES).to(DEVICE)
        self.opt   = torch.optim.SGD(self.model.parameters(), lr=LR)

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, params, config=None):
        sd = {k: torch.as_tensor(p) for k, p in
              zip(self.model.state_dict().keys(), params)}
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

# Uncomment for single class in all the clients
# TARGET_LABEL      = 0        # <<<  choose the single class all clients use


# def build_client_dataset(cid: int):
#     aux = torch.load(AUX_PATH, weights_only=False)

#     label_nodes = (aux.y == TARGET_LABEL).nonzero(as_tuple=False).view(-1)

#     # deterministic shuffle once (stored in file so every run is identical)
#     perm_path = f"aux_perm_label{TARGET_LABEL}.pt"
#     if os.path.exists(perm_path):
#         perm = torch.load(perm_path)
#     else:
#         perm = label_nodes[torch.randperm(len(label_nodes))]
#         torch.save(perm, perm_path)

#     # split that permutation into NUM_CLIENTS roughly-equal chunks
#     chunk_size = int(np.ceil(len(perm) / NUM_CLIENTS))
#     chunk = perm[cid*chunk_size : (cid+1)*chunk_size]
#     if len(chunk) == 0:
#         raise ValueError(
#             f"Client {cid} got zero samples; "
#             f"reduce NUM_CLIENTS or use a label with more examples."
#         )

#     # induce sub-graph on those nodes
#     eidx, _ = subgraph(chunk, aux.edge_index, relabel_nodes=True,
#                        num_nodes=aux.num_nodes)

#     d = Data(x=aux.x[chunk],
#              y=aux.y[chunk],
#              edge_index=eidx)

#     # one-hot label distribution: only TARGET_LABEL is present
#     dist = np.zeros(GLOBAL_NUM_CLASSES, dtype=np.float32)
#     dist[TARGET_LABEL] = 1.0
#     out_dir = f"shadow_logs/FL_c{TARGET_LABEL}"
#     os.makedirs(out_dir, exist_ok=True)
#     with open(os.path.join(out_dir, f"label_dist_client{cid}.pkl"), "wb") as f:
#         pickle.dump(dist, f)

#     train_idx = torch.arange(d.num_nodes)
#     return d.to(DEVICE), train_idx.to(DEVICE)
# random.seed(42)
# Choose which label the special client will have, and which client ID is special
TARGET_LABEL  = 8   # Selection of class (0–8)
SPECIAL_CID   = 4   # Choose which client to have all samples

def build_client_dataset(cid: int):
    aux       = torch.load(AUX_PATH, weights_only=False)
    num_nodes = aux.num_nodes

    if cid == SPECIAL_CID:
        # collect every node of the target label
        nodes = (aux.y == TARGET_LABEL).nonzero(as_tuple=False).view(-1)

        # deterministic shuffle
        perm_path = f"aux_perm_label{TARGET_LABEL}.pt"
        if os.path.exists(perm_path):
            perm = torch.load(perm_path)
        else:
            perm = nodes[torch.randperm(len(nodes))]
            torch.save(perm, perm_path)
        chunk = perm

        # build one‐hot distribution
        dist = np.zeros(GLOBAL_NUM_CLASSES, dtype=np.float32)
        dist[TARGET_LABEL] = 1.0
        out_dir = f"shadow_logs/FLsingleL{TARGET_LABEL}C{SPECIAL_CID}"

    else:
        # shuffle ALL nodes once
        all_nodes = torch.arange(num_nodes)
        perm_path = "aux_perm_all_nodes.pt"
        if os.path.exists(perm_path):
            perm = torch.load(perm_path)
        else:
            perm = all_nodes[torch.randperm(num_nodes)]
            torch.save(perm, perm_path)

        # chunk size = ceil(total_nodes / NUM_CLIENTS)
        chunk_size = int(math.ceil(num_nodes / NUM_CLIENTS))
        chunk = perm[cid * chunk_size : (cid + 1) * chunk_size]

        # empirical label distribution on this chunk
        labels_np = aux.y[chunk].cpu().numpy()
        counts    = np.bincount(labels_np, minlength=GLOBAL_NUM_CLASSES)
        dist      = counts.astype(np.float32) / counts.sum()
        out_dir   = f"shadow_logs/FLsingleL{TARGET_LABEL}C{SPECIAL_CID}"

    # induce the subgraph on `chunk`
    eidx, _ = subgraph(chunk, aux.edge_index, relabel_nodes=True, num_nodes=num_nodes)
    d       = Data(x=aux.x[chunk], y=aux.y[chunk], edge_index=eidx)

    # save the computed label distribution
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"label_dist_client{cid}.pkl"), "wb") as f:
        pickle.dump(dist, f)

    train_idx = torch.arange(d.num_nodes)
    return d.to(DEVICE), train_idx.to(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID 0-9")
    args = parser.parse_args()

    data, idx = build_client_dataset(args.cid)
    client    = AuxClient(args.cid, data, idx)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
