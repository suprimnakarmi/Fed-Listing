from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import torch
import numpy as np

# def split_communities(data,num_client,rand_seed):
    
#     G = to_networkx(data, to_undirected=True, node_attrs=['x','y'])
#     communities = sorted(nx.community.asyn_fluidc(G, num_client, max_iter = 5000, seed= rand_seed))
#     # communities = sorted(nx.community.asyn_fluidc(G, num_client, max_iter = 50, seed= 0))

#     node_groups = []
#     for com in communities:
#         node_groups.append(list(com))   
#     list_of_clients = []

#     for i in range(num_client):
#         print("Yeta")
#         print("Client no.: ", num_client)
#         list_of_clients.append(from_networkx(G.subgraph(node_groups[i]).copy()))

#     return list_of_clients

# def split_dataset(data,split_percentage):
#     mask = torch.randn((data.num_nodes)) < split_percentage
#     nmask = torch.logical_not(mask)

#     train_mask = mask
#     test_mask = nmask
#     data.train_mask = train_mask
#     data.test_mask = test_mask
#     if test_mask.numel()==0:
#         print('no enough test data!')
#         exit()
#     return data


# def process_dataset(dataset,num_clients,split):

#     print(f'Dataset: {dataset}:')
#     print('======================')
#     print(f'Number of graphs: {len(dataset)}')
#     print(f'Number of features: {dataset.num_features}')
#     print(f'Number of classes: {dataset.num_classes}')
#     print(f"Number of clients: {num_clients}")
#     print(f"Type of client: {type(num_clients)}")

#     data = dataset[0]  # Get the graph object.

#     print(data)
#     print('==============================================================')

#     # Gather some statistics about the graph.
#     print(f'Number of nodes: {data.num_nodes}')
#     print(f'Number of edges: {data.num_edges}')

#     client_data =  split_communities(data,num_clients)

#     for k in range(len(client_data)):
#         client_data[k]=split_dataset(client_data[k], split)

#     return client_data,dataset.num_classes

from typing import List
import torch
import numpy as np
from torch_geometric.data import Data

# ----------------------------
# Helpers (match your big script)
# ----------------------------

def _by_class(data: Data, idx: torch.Tensor, c: int) -> torch.Tensor:
    """Filter indices in idx that belong to class c."""
    return idx[data.y[idx] == c]

def _cap(idx: torch.Tensor, k: int | None) -> torch.Tensor:
    if k is None or len(idx) <= k:
        return idx
    return idx[:k]

def _balanced_subset(data: Data, idx: torch.Tensor, num_classes: int, per_class: int | None):
    """Take a balanced subset from idx across all classes."""
    parts = []
    for c in range(num_classes):
        ic = _by_class(data, idx, c)
        parts.append(ic)
    if per_class is None:
        per_class = min(len(ic) for ic in parts) if parts else 0
    if per_class == 0:
        return torch.tensor([], dtype=torch.long)
    kept = [_cap(ic, per_class) for ic in parts]
    return torch.cat(kept) if kept else torch.tensor([], dtype=torch.long)

def _dshuf(idx: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return idx[torch.randperm(len(idx), generator=g)] if len(idx) > 0 else idx

# ----------------------------
# Main API
# ----------------------------

def split_communities(data: Data, num_client: int, rand_seed: int) -> List[Data]:
    """
    Return per-client shallow copies of the SAME graph.
    We keep the whole graph per client and later set client-specific masks.
    """
    clients = []
    for _ in range(num_client):
        # shallow copy is fine (x, y, edge_index shared), masks will differ:
        clients.append(data.clone())
    return clients

def split_dataset(data: Data, client_id: int, num_classes: int, test_fraction: float,
                  num_clients: int | None = None, rand_seed: int = 42) -> Data:
    """
    Assign train/test masks so that the client's train set follows your
    previously-defined label-split behavior.

    Parameters
    ----------
    data : Data
        Full Cora graph (shared across clients).
    client_id : int
        ID of this client.
    num_classes : int
        Number of classes in dataset.
    test_fraction : float
        Fraction of nodes used for testing (shared across all clients).
    num_clients : int | None
        Total number of clients (needed to chunk the training pool).
        If None, we infer from data.train_mask/test_mask if possible; otherwise default 10.
    rand_seed : int
        Seed for deterministic splits.
    """
    # Total clients default (kept consistent with your previous code default=10)
    if num_clients is None:
        num_clients = 10

    N = data.num_nodes
    assert isinstance(N, int) and N > 0, "Data must contain nodes."

    # ----------------------------
    # Build shared test/train pool
    # ----------------------------
    g = torch.Generator().manual_seed(rand_seed)
    perm_all = torch.arange(N, dtype=torch.long)[torch.randperm(N, generator=g)]
    num_test = int(round(float(N) * float(test_fraction)))
    num_test = max(1, min(N - 1, num_test))  # keep at least 1 test and 1 train

    test_pool = perm_all[:num_test]
    train_pool = perm_all[num_test:]

    # Shared masks for this client's Data view
    test_mask = torch.zeros(N, dtype=torch.bool)
    test_mask[test_pool] = True

    # ----------------------------
    # Disjoint "base buckets" from the train pool (global, deterministic)
    # ----------------------------
    g_global = torch.Generator().manual_seed(42)  # fixed global seed like your script
    perm_train = train_pool[torch.randperm(len(train_pool), generator=g_global)]
    base_splits = torch.chunk(perm_train, num_clients)
    # Guard: if num_clients > len(perm_train), chunk may produce smaller last chunks. That's okay.
    base_idx = base_splits[client_id].clone() if client_id < len(base_splits) else torch.tensor([], dtype=torch.long)

    # Target size (for optional capping)
    target_size = len(train_pool) // num_clients if num_clients > 0 else len(train_pool)

    # ----------------------------
    # Client-specific label-split logic (exactly your scenarios)
    # ----------------------------
    if client_id == 0:
        # Balanced within own bucket (no overlap with others)
        local_train_idx = _balanced_subset(data, base_idx, num_classes, per_class=None)
        local_train_idx = _dshuf(local_train_idx, seed=42 + client_id)

    elif client_id in {1, 2, 3, 5, 6}:
        # IID-ish: just keep the base bucket
        local_train_idx = _dshuf(base_idx, seed=42 + client_id)

    elif client_id == 4:
        # Missing class 3
        keep_idx = base_idx[data.y[base_idx] != 2]
        local_train_idx = _cap(_dshuf(keep_idx, seed=42 + client_id), target_size)

    elif client_id == 7:
        # Only class 2
        only_c2 = _by_class(data, base_idx, 2)
        local_train_idx = _dshuf(only_c2, seed=42 + client_id)

    else:
        # 80/20 dominant-class skew within this client's bucket
        if len(base_idx) == 0:
            local_train_idx = base_idx
        else:
            y_local = data.y[base_idx]
            counts_local = torch.bincount(y_local, minlength=num_classes)
            dom = int(counts_local.argmax().item())

            total = min(target_size, len(base_idx)) if target_size > 0 else len(base_idx)
            n_dom = int(0.8 * total)
            n_rem = max(0, total - n_dom)

            dom_idx = _dshuf(_by_class(data, base_idx, dom), seed=777 + client_id)
            other_all = torch.cat([_by_class(data, base_idx, c) for c in range(num_classes) if c != dom]) \
                        if num_classes > 1 else torch.tensor([], dtype=torch.long)
            other_idx = _dshuf(other_all, seed=1234 + client_id)

            dom_take = _cap(dom_idx, n_dom)
            rem_take = _cap(other_idx, n_rem)
            if len(dom_take) + len(rem_take) > 0:
                local_train_idx = torch.cat([dom_take, rem_take])
                local_train_idx = _dshuf(local_train_idx, seed=999 + client_id)
            else:
                local_train_idx = base_idx

    # ----------------------------
    # Assign masks to this client's Data view
    # ----------------------------
    train_mask = torch.zeros(N, dtype=torch.bool)
    if len(local_train_idx) > 0:
        train_mask[local_train_idx] = True

    data.train_mask = train_mask
    data.test_mask = test_mask
    return data
