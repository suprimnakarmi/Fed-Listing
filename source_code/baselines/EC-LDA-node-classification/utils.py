from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj ,dense_to_sparse
import numpy as np
import random
from torch_geometric.utils import is_undirected

def extract_node_subgraphs(graph: Data, num_hops,device):
    """
    Extract enclosing subgraphs for every node in the graph.

    Args:
    :param graph : Data contains attributes edge_index, x, y
    :param num_hops : number of hops
    """
    data_list = []
    # node_index = graph.edge_index.unique()
    # for src in node_index.t().tolist():  # The times of loop execution = #links = #samples
    # assert sorted(graph.edge_index.unique().tolist()) == list(range(graph.x.shape[0])), "node index does not match to node nums!" TODO
    for src in range(graph.x.shape[0]):
        src_origin = src
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src], num_hops, graph.edge_index,
            relabel_nodes=True)  # relabel Ture, label index in sub_edge_index will change.
        data = Data(x=graph.x[sub_nodes], src=src_origin, edge_index=sub_edge_index, y=graph.y[src],
                    sub_nodes=sub_nodes).to(device)  # sub nodes are original index.
        # if dataset_name == "cora":
        #     data["train_mask"] = graph.train_mask[src]
        #     data["val_mask"] = graph.val_mask[src]
        #     data["test_mask"] = graph.test_mask[src]
        data_list.append([data,mapping])
    return data_list


def DP_clip_edges(edge_index,k,num_nodes):
    undirected = is_undirected(edge_index)
    adj = to_dense_adj(edge_index)
    
    for i in range(num_nodes):
        
        links = []
        for j in range(num_nodes):
            if j==i:
                continue
            if adj[0][j][i]==1:
                links.append(j)
        
        if len(links)>k:
            remove_links = random.sample(links, len(links)-k)
            for j in remove_links:
                if undirected:
                    adj[0][i][j]=0
                    adj[0][j][i]=0
                else:
                    adj[0][j][i]=0
    
    return dense_to_sparse(adj)
