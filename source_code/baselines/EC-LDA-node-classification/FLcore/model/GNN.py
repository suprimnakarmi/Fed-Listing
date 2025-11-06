from torch_geometric.nn import GCNConv, SAGEConv,GATConv, GINConv
import torch.nn.functional as F
import torch
from torch import nn


class GCN(torch.nn.Module):
    # def __init__(self,hops, hidden_channels1,hidden_channels2,hidden_channels3,features_in, features_out):
    def __init__(self,dataset,hops,features_in, features_out,c4096=4096,c1024=1024,c512=512,c256=256,c128=128,c64=64,c16=16):
        super().__init__()
        # self.conv1 = GCNConv(features_in, hidden_channels2)
        # self.conv2 = GCNConv(hidden_channels2, hidden_channels3)
        # self.conv3 = GCNConv(hidden_channels3, hidden_channels3)
        # # self.fc1 = nn.Linear(hidden_channels2,hidden_channels3)
        self.hops = hops
        # # self.activation = nn.LeakyReLU()
        # # self.activation = nn.ELU()
        if dataset in ['Cora','CiteSeer']:
            if hops == 1:
                self.conv1 = GCNConv(features_in,64)
            elif hops == 2:
                self.conv1 = GCNConv(features_in,512)
                self.conv2 = GCNConv(512,64)
            elif hops == 3:
                self.conv1 = GCNConv(features_in,512)
                self.conv2 = GCNConv(512,256)
                self.conv3 = GCNConv(256,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset in ['PubMed','FacebookPagePage','WikiCS','AmazonComputers']:
            if hops == 1:
                self.conv1 = GCNConv(features_in,64)
            elif hops == 2:
                self.conv1 = GCNConv(features_in,128)
                self.conv2 = GCNConv(128,64)
            elif hops == 3:
                self.conv1 = GCNConv(features_in,128)
                self.conv2 = GCNConv(128,128)
                self.conv3 = GCNConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'CoraFull':
            if hops == 1:
                self.conv1 = GCNConv(features_in,64)
            elif hops == 2:
                self.conv1 = GCNConv(features_in,512)
                self.conv2 = GCNConv(512,64)
            elif hops == 3:
                self.conv1 = GCNConv(features_in,1024)
                self.conv2 = GCNConv(1024,128)
                self.conv3 = GCNConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset in ['Reddit2']:
            if hops == 1:
                self.conv1 = GCNConv(features_in,64)
            elif hops == 2:
                self.conv1 = GCNConv(features_in,256)
                self.conv2 = GCNConv(256,64)
            elif hops == 3:
                self.conv1 = GCNConv(features_in,256)
                self.conv2 = GCNConv(256,128)
                self.conv3 = GCNConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'LastFM':
            if hops == 1:
                self.conv1 = GCNConv(features_in,64)
            elif hops == 2:
                self.conv1 = GCNConv(features_in,64)
                self.conv2 = GCNConv(64,64)
            elif hops == 3:
                self.conv1 = GCNConv(features_in,128)
                self.conv2 = GCNConv(128,64)
                self.conv3 = GCNConv(64,64)
            else:
                print("Wrong hops!")
                exit()

        if dataset == 'CoraFull':
            self.fc2 = nn.Linear(64,features_out)
        else:
            self.fc2 = nn.Linear(64,features_out)
        
        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()
        # self.activation = nn.LeakyReLU()

    def forward(self, x, edge_index):
        if self.hops == 1:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
        elif self.hops == 2:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
        else:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
            x = self.conv3(x,edge_index)
            x = self.activation(x)

        emb = x.clone().detach()
        x = self.fc2(x)
        # x = x.relu()
        # return x, torch.var(emb.sum(dim=1))
        return x, emb
    
class GAT(torch.nn.Module):
    def __init__(self,dataset,hops,features_in, features_out,c4096=4096,c1024=1024,c512=512,c256=256,c128=128,c64=64,c16=16):
        super().__init__()
        # self.conv1 = GCNConv(features_in, hidden_channels2)
        # self.conv2 = GCNConv(hidden_channels2, hidden_channels3)
        # self.conv3 = GCNConv(hidden_channels3, hidden_channels3)
        # # self.fc1 = nn.Linear(hidden_channels2,hidden_channels3)
        self.hops = hops
        
        # # self.activation = nn.LeakyReLU()
        # # self.activation = nn.ELU()
        if dataset in ['Cora','CiteSeer']:
            if hops == 1:
                self.conv1 = GATConv(features_in,64)
            elif hops == 2:
                self.conv1 = GATConv(features_in,512)
                self.conv2 = GATConv(512,64)
            elif hops == 3:
                self.conv1 = GATConv(features_in,512)
                self.conv2 = GATConv(512,256)
                self.conv3 = GATConv(256,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset in ['PubMed','FacebookPagePage','WikiCS','AmazonComputers']:
            if hops == 1:
                self.conv1 = GATConv(features_in,64)
            elif hops == 2:
                self.conv1 = GATConv(features_in,128)
                self.conv2 = GATConv(128,64)
            elif hops == 3:
                self.conv1 = GATConv(features_in,128)
                self.conv2 = GATConv(128,128)
                self.conv3 = GATConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'CoraFull':
            if hops == 1:
                self.conv1 = GATConv(features_in,64)
            elif hops == 2:
                self.conv1 = GATConv(features_in,1024)
                self.conv2 = GATConv(1024,64)
            elif hops == 3:
                self.conv1 = GATConv(features_in,1024)
                self.conv2 = GATConv(1024,128)
                self.conv3 = GATConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'Reddit2':
            if hops == 1:
                self.conv1 = GATConv(features_in,64)
            elif hops == 2:
                self.conv1 = GATConv(features_in,256)
                self.conv2 = GATConv(256,64)
            elif hops == 3:
                self.conv1 = GATConv(features_in,256)
                self.conv2 = GATConv(256,128)
                self.conv3 = GATConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'LastFM':
            if hops == 1:
                self.conv1 = GATConv(features_in,64)
            elif hops == 2:
                self.conv1 = GATConv(features_in,64)
                self.conv2 = GATConv(64,64)
            elif hops == 3:
                self.conv1 = GATConv(features_in,128)
                self.conv2 = GATConv(128,64)
                self.conv3 = GATConv(64,64)
            else:
                print("Wrong hops!")
                exit()

        self.fc2 = nn.Linear(64,features_out)
        
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        if self.hops == 1:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
        elif self.hops == 2:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
        else:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
            x = self.conv3(x,edge_index)
            x = self.activation(x)

        emb = x.clone().detach()
        x = self.fc2(x)
        # x = x.relu()
        # return x, torch.var(emb.sum(dim=1))
        return x, emb
    

class GraphSAGE(torch.nn.Module):
    def __init__(self,dataset,hops,features_in, features_out,c4096=4096,c1024=1024,c512=512,c256=256,c128=128,c64=64,c16=16):
        super().__init__()
        # self.conv1 = GCNConv(features_in, hidden_channels2)
        # self.conv2 = GCNConv(hidden_channels2, hidden_channels3)
        # self.conv3 = GCNConv(hidden_channels3, hidden_channels3)
        # # self.fc1 = nn.Linear(hidden_channels2,hidden_channels3)
        self.hops = hops
        
        # # self.activation = nn.LeakyReLU()
        # # self.activation = nn.ELU()
        if dataset in ['Cora','CiteSeer']:
            if hops == 1:
                self.conv1 = SAGEConv(features_in,64)
            elif hops == 2:
                self.conv1 = SAGEConv(features_in,512)
                self.conv2 = SAGEConv(512,64)
            elif hops == 3:
                self.conv1 = SAGEConv(features_in,512)
                self.conv2 = SAGEConv(512,256)
                self.conv3 = SAGEConv(256,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset in ['PubMed','FacebookPagePage','WikiCS','AmazonComputers']:
            if hops == 1:
                self.conv1 = SAGEConv(features_in,64)
            elif hops == 2:
                self.conv1 = SAGEConv(features_in,128)
                self.conv2 = SAGEConv(128,64)
            elif hops == 3:
                self.conv1 = SAGEConv(features_in,128)
                self.conv2 = SAGEConv(128,128)
                self.conv3 = SAGEConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'CoraFull':
            if hops == 1:
                self.conv1 = SAGEConv(features_in,64)
            elif hops == 2:
                self.conv1 = SAGEConv(features_in,1024)
                self.conv2 = SAGEConv(1024,64)
            elif hops == 3:
                self.conv1 = SAGEConv(features_in,1024)
                self.conv2 = SAGEConv(1024,128)
                self.conv3 = SAGEConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'Reddit2':
            if hops == 1:
                self.conv1 = SAGEConv(features_in,64)
            elif hops == 2:
                self.conv1 = SAGEConv(features_in,256)
                self.conv2 = SAGEConv(256,64)
            elif hops == 3:
                self.conv1 = SAGEConv(features_in,256)
                self.conv2 = SAGEConv(256,128)
                self.conv3 = SAGEConv(128,64)
            else:
                print("Wrong hops!")
                exit()
        elif dataset == 'LastFM':
            if hops == 1:
                self.conv1 = SAGEConv(features_in,64)
            elif hops == 2:
                self.conv1 = SAGEConv(features_in,64)
                self.conv2 = SAGEConv(64,64)
            elif hops == 3:
                self.conv1 = SAGEConv(features_in,128)
                self.conv2 = SAGEConv(128,64)
                self.conv3 = SAGEConv(64,64)
            else:
                print("Wrong hops!")
                exit()
                
        self.fc2 = nn.Linear(64,features_out)
        
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_index):
        if self.hops == 1:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
        elif self.hops == 2:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
        else:
            x = self.conv1(x,edge_index)
            x = self.activation(x)
            x = self.conv2(x,edge_index)
            x = self.activation(x)
            x = self.conv3(x,edge_index)
            x = self.activation(x)

        emb = x.clone().detach()
        x = self.fc2(x)
        # x = x.relu()
        # return x, torch.var(emb.sum(dim=1))
        return x, emb

    
class GIN(torch.nn.Module):
    def __init__(self, dataset, hops, features_in, features_out,
                 c4096=4096, c1024=1024, c512=512, c256=256, c128=128, c64=64, c16=16):
        super().__init__()
        self.hops = hops

        def gin_mlp(in_c, out_c):
            # 2-layer MLP as in the GIN paper
            return nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.ReLU(),
                nn.Linear(out_c, out_c),
            )

        # Select layer widths by dataset (mirrors your GCN choices)
        if dataset in ['Cora', 'CiteSeer']:
            if hops == 1:
                self.conv1 = GINConv(gin_mlp(features_in, 64), train_eps=True)
            elif hops == 2:
                self.conv1 = GINConv(gin_mlp(features_in, 512), train_eps=True)
                self.conv2 = GINConv(gin_mlp(512, 64), train_eps=True)
            elif hops == 3:
                self.conv1 = GINConv(gin_mlp(features_in, 512), train_eps=True)
                self.conv2 = GINConv(gin_mlp(512, 256), train_eps=True)
                self.conv3 = GINConv(gin_mlp(256, 64), train_eps=True)
            else:
                print("Wrong hops!"); exit()

        elif dataset in ['PubMed', 'FacebookPagePage', 'WikiCS', 'AmazonComputers']:
            if hops == 1:
                self.conv1 = GINConv(gin_mlp(features_in, 64), train_eps=True)
            elif hops == 2:
                self.conv1 = GINConv(gin_mlp(features_in, 128), train_eps=True)
                self.conv2 = GINConv(gin_mlp(128, 64), train_eps=True)
            elif hops == 3:
                self.conv1 = GINConv(gin_mlp(features_in, 128), train_eps=True)
                self.conv2 = GINConv(gin_mlp(128, 128), train_eps=True)
                self.conv3 = GINConv(gin_mlp(128, 64), train_eps=True)
            else:
                print("Wrong hops!"); exit()

        elif dataset == 'CoraFull':
            if hops == 1:
                self.conv1 = GINConv(gin_mlp(features_in, 64), train_eps=True)
            elif hops == 2:
                self.conv1 = GINConv(gin_mlp(features_in, 512), train_eps=True)
                self.conv2 = GINConv(gin_mlp(512, 64), train_eps=True)
            elif hops == 3:
                self.conv1 = GINConv(gin_mlp(features_in, 1024), train_eps=True)
                self.conv2 = GINConv(gin_mlp(1024, 128), train_eps=True)
                self.conv3 = GINConv(gin_mlp(128, 64), train_eps=True)
            else:
                print("Wrong hops!"); exit()

        elif dataset in ['Reddit2']:
            if hops == 1:
                self.conv1 = GINConv(gin_mlp(features_in, 64), train_eps=True)
            elif hops == 2:
                self.conv1 = GINConv(gin_mlp(features_in, 256), train_eps=True)
                self.conv2 = GINConv(gin_mlp(256, 64), train_eps=True)
            elif hops == 3:
                self.conv1 = GINConv(gin_mlp(features_in, 256), train_eps=True)
                self.conv2 = GINConv(gin_mlp(256, 128), train_eps=True)
                self.conv3 = GINConv(gin_mlp(128, 64), train_eps=True)
            else:
                print("Wrong hops!"); exit()

        elif dataset == 'LastFM':
            if hops == 1:
                self.conv1 = GINConv(gin_mlp(features_in, 64), train_eps=True)
            elif hops == 2:
                self.conv1 = GINConv(gin_mlp(features_in, 64), train_eps=True)
                self.conv2 = GINConv(gin_mlp(64, 64), train_eps=True)
            elif hops == 3:
                self.conv1 = GINConv(gin_mlp(features_in, 128), train_eps=True)
                self.conv2 = GINConv(gin_mlp(128, 64), train_eps=True)
                self.conv3 = GINConv(gin_mlp(64, 64), train_eps=True)
            else:
                print("Wrong hops!"); exit()
        else:
            print("Unknown dataset!"); exit()

        # Final classifier head (same as your GCN: node-wise logits)
        self.fc2 = nn.Linear(64, features_out)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        if self.hops == 1:
            x = self.activation(self.conv1(x, edge_index))
        elif self.hops == 2:
            x = self.activation(self.conv1(x, edge_index))
            x = self.activation(self.conv2(x, edge_index))
        else:  # hops == 3
            x = self.activation(self.conv1(x, edge_index))
            x = self.activation(self.conv2(x, edge_index))
            x = self.activation(self.conv3(x, edge_index))

        emb = x.clone().detach()
        x = self.fc2(x)
        return x, emb
