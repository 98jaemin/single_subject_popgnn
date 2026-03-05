import torch
from torch import nn 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils  import get_global_edge_inputs

class PopulationGNN(nn.Module):
    def __init__(self, in_dim=2000, hid_dim=20, num_layers=4, dropout=0.3):
        super(PopulationGNN, self).__init__()
        
        self.drop_p = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            if i > 0:
                self.convs.append(GCNConv(hid_dim, hid_dim))
                self.bns.append(nn.BatchNorm1d(hid_dim))
            else:    # i == 0
                self.convs.append(GCNConv(in_dim, hid_dim))
                self.bns.append(nn.BatchNorm1d(hid_dim))
                
        self.weights = nn.Parameter(torch.randn(num_layers))   # AWAB: weights for learnable readout
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.projection.reset_parameters()
        nn.init.normal_(self.weights)

    def forward(self, features, edges, edge_weight=None):
        x = features 
        layer_out = []
        
        for i in range(len(self.convs)):
            if i > 0:
                x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.convs[i](x, edges, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            if i > 0:
                x = x + 0.7 * layer_out[i - 1]
            layer_out.append(x)

        weight = F.softmax(self.weights, dim=0)
        for i in range(len(layer_out)):
            layer_out[i] = layer_out[i] * weight[i]
        emb = sum(layer_out)
        return emb   # shape : (num_nodes, hid_dim)

class EDGE(nn.Module):
    def __init__(self, input_dim, hid_dim=128, dropout=0.2):
        super(EDGE, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(input_dim, hid_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hid_dim),
                nn.Dropout(dropout),
                nn.Linear(hid_dim, hid_dim, bias=True),
                )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.input_dim = input_dim
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
                    
    def forward(self, x):
        x1 = x[:, 0:self.input_dim]
        x2 = x[:, self.input_dim:]
        
        h1 = self.mlp(x1) 
        h2 = self.mlp(x2) 
        p = (self.cos(h1, h2) + 1) * 0.5
        return p
            
class IPGNN(nn.Module):
    def __init__(
        self, device, in_dim=200, num_pheno_features=2,
        hid_dim_gnn=16, num_layers=4, dropout=0.3, num_classes=2, global_edge_thr=1.1,
        hid_dim_edge=128, dropout_edge=0.2,
    ):
        super(IPGNN, self).__init__()

        self.device = device
        gnn_in_dim = int(in_dim * (in_dim - 1) / 2)
        
        self.pop_gnn = PopulationGNN(in_dim=gnn_in_dim, hid_dim=hid_dim_gnn, num_layers=num_layers, dropout=dropout)
        self.global_edge_thr = global_edge_thr
        
        self.edge_mlp = EDGE(num_pheno_features, hid_dim=hid_dim_edge, dropout=dropout_edge)  
        self.projection_head = nn.Linear(hid_dim_gnn, num_classes)
        
        
    def forward(self, embeddings: torch.Tensor, phenotypic_dict: dict):        
        edge_index, edge_input = get_global_edge_inputs(embeddings, phenotypic_dict, self.global_edge_thr)
        edge_index = torch.LongTensor(edge_index).to(self.device)
        edge_input = (edge_input - edge_input.mean(axis=0)) / edge_input.std(axis=0)
        edge_input = torch.FloatTensor(edge_input).to(self.device)
        edge_weight = torch.squeeze(self.edge_mlp(edge_input))
        
        node_embs = self.pop_gnn(embeddings, edge_index, edge_weight)    # shape : (num_nodes, hid_dim_gnn)
        predictions = self.projection_head(node_embs)                    # shape : (num_nodes, num_classes)
        
        return predictions, edge_index.shape[1]

