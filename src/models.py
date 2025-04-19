import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# NN model
class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.lin1(x).relu()
        h = self.lin2(h).relu()

        output = self.out_layer(h).squeeze()
        return output
    
# GCN model
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
       
        out = self.classifier(h).squeeze()
        return out



# GCN+LSTM model
class GCN_LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_index):
        super().__init__()
        torch.manual_seed(1234)
        self.edge_weight = torch.nn.Parameter(torch.zeros(edge_index.shape[1]))

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.conv3 = GRUConv(hidden_dim, hidden_dim)  # Removed as GRUConv is not defined
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index, torch.exp(self.edge_weight)).relu()
        h = self.conv2(h, edge_index, torch.exp(self.edge_weight)).relu()
        # h = self.conv3(h, edge_index, torch.exp(self.edge_weight)).relu()
        out = self.classifier(h).squeeze()
        return out

# GCN+BiGRU model
class GCN_BiGRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        #h = h.unsqueeze(0)  # Add batch dimension for GRU
        h, _ = self.gru1(h)
        h, _ = self.gru2(h)
        h = h.squeeze(0)  # Remove batch dimension after GRU
        out = self.classifier(h).squeeze()
        return out
