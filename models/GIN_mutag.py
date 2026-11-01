import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np
import os
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool
class GAT(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 output_dim=2,
                 num_heads=8,
                 dropout=0.6,
                 negative_slope=0.2,
                 residual=True):
        super().__init__()
        out_channels = output_dim

        self.conv1 = GATConv(
            in_channels, 
            hidden_channels, 
            heads=num_heads, 
            dropout=dropout,
            negative_slope=negative_slope,
            concat=True
        )
        

        self.conv2 = GATConv(
            hidden_channels * num_heads,
            hidden_channels, 
            heads=1,
            dropout=dropout,
            negative_slope=negative_slope,
            concat=False
        )
        
        self.drop_out = dropout
        self.negative_slope = negative_slope
        self.residual = residual
        

        if residual:
            self.residual_linear = nn.Linear(in_channels, hidden_channels * num_heads)
        else:
            self.register_parameter('residual_linear', None)
        self.classifier1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),)
        self.classifier2 = nn.Sequential(
            nn.Linear(hidden_channels, out_channels)
        )
    def forward(self, x, edge_index, batch, activate_num=0):

        x_initial = x
        count = 0
        self.Dropout = nn.Dropout(self.drop_out)

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        

        if self.residual and self.residual_linear is not None:
            res = self.residual_linear(x_initial)
            x = x + res
        
        x = self.Dropout(x)


        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.classifier1(x)
        size1 = x.shape[0] * x.shape[1] 
        count += (x > 0).sum().item()
        activate_num += count / size1
        x = F.relu(x)
        x = self.classifier2(x)
        return  x , activate_num


class DeepGAT(GAT):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_heads=8, 
                 dropout=0.6,
                 negative_slope=0.2,
                 residual=True):
        super().__init__(in_channels, hidden_channels, out_channels, num_heads, dropout, negative_slope, residual)
        

        self.mid_conv = GATConv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            negative_slope=negative_slope,
            concat=True
        )
        

        if residual:
            self.mid_residual = nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
        else:
            self.register_parameter('mid_residual', None)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_initial = x
        

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        

        if self.residual and self.residual_linear is not None:
            res = self.residual_linear(x_initial)
            x = x + res
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        

        x_mid = x
        x = self.mid_conv(x, edge_index)
        x = F.elu(x)
        

        if self.residual and self.mid_residual is not None:
            res = self.mid_residual(x_mid)
            x = x + res
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        

        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
class GIN(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, drop_out=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.Relu = nn.ReLU()
        self.drop_out = drop_out
        self.Dropout = nn.Dropout(drop_out)

        self.convs1 = torch.nn.ModuleList()
        self.convs2 = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU())
            self.convs1.append(GINConv(mlp, train_eps=True))

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim)
            )
            self.convs2.append(GINConv(mlp, train_eps=True))


        self.classifier1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),)
        self.classifier2 = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        

        self.initial_lin = nn.Linear(7, hidden_dim)
    
    def forward(self, x, edge_index, batch, activate_num=0):
        count = 0
        self.Dropout = nn.Dropout(self.drop_out)

        x = self.initial_lin(x)
        for i in range(self.num_layers):
            x_res = x
            x = self.convs1[i](x,edge_index)
            x = self.Dropout(x)
            x = self.convs2[i](x,edge_index)
            count += (x > 0).sum().item()

            x = F.relu(x)+x_res     

        size1 = x.shape[0]*x.shape[1]           

        x = global_add_pool(x, batch)
        

        x = self.classifier1(x)
        x = self.Relu(x)
        size2 = x.shape[0]*x.shape[1]
        count += (x > 0).sum().item()

        
        activate_num+=count / (self.num_layers * size1 + size2)

        x = self.Dropout(x)
        x = self.classifier2(x)


        return x.squeeze(),activate_num