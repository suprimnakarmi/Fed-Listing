from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch
from dataset.utils.dataset_utils import split_communities,split_dataset

# def generate_PubMed(num_clients,split,rand_seed):

#     dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=T.LargestConnectedComponents())

#     print(f'Dataset: {dataset}:')
#     print('======================')
#     print(f'Number of graphs: {len(dataset)}')
#     print(f'Number of features: {dataset.num_features}')
#     print(f'Number of classes: {dataset.num_classes}')

#     data = dataset[0]  # Get the graph object.

#     print(data)
#     print('==============================================================')

#     # Gather some statistics about the graph.
#     print(f'Number of nodes: {data.num_nodes}')
#     print(f'Number of edges: {data.num_edges}')

#     client_data =  split_communities(data,num_clients,rand_seed)

#     for k in range(len(client_data)):
#         client_data[k]=split_dataset(client_data[k], split)

#     return client_data,dataset.num_classes

def generate_PubMed(num_clients: int, split: float, rand_seed: int):
    """
    Generate per-client Data objects for Cora with client-specific label splits
    that match your existing logic.

    Args
    ----
    num_clients : int
        Total number of clients.
    split : float
        Test fraction (e.g., 0.1 means 10% nodes as test for everyone).
    rand_seed : int
        Seed for deterministic behavior.

    Returns
    -------
    client_data : list[Data]
        One Data object per client (same graph, different masks).
    num_classes : int
        Number of classes in Cora.
    """
    dataset = Planetoid(root='data/Planetoid', name='PubMed',
                        transform=T.LargestConnectedComponents())

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f"Number of clients: {num_clients}")
    print(f"Type of client: {type(num_clients)}")

    data = dataset[0]  # The graph object

    print(data)
    print('==============================================================')

    # Graph stats
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')

    # Create per-client shallow copies of the SAME graph
    client_data = split_communities(data, num_clients, rand_seed)
    print(f"Client data: {len(client_data)}")
    print(f"Client data type: {type(client_data)}")

    # For each client, assign masks according to your per-client label-split logic
    for k in range(len(client_data)):
        client_data[k] = split_dataset(
            client_data[k],
            client_id=k,
            num_classes=dataset.num_classes,
            test_fraction=split,
            num_clients=num_clients,
            rand_seed=rand_seed
        )

    return client_data, dataset.num_classes


