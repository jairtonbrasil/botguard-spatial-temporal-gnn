import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class TemporalEncoder(nn.Module):
    """
    Bi-directional GRU to process the sequence of recent user actions.
    Captures temporal entropy and action frequency.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(TemporalEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.gru(x)
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        return torch.cat((forward_hidden, backward_hidden), dim=1)

class SpatialEncoder(nn.Module):
    """
    GraphSAGE to process the user's neighborhood topology.
    Captures follower/following asymmetry and isolation.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(SpatialEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x shape: (num_nodes, in_channels)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class HybridBotDetector(nn.Module):
    """
    Combines Spatial and Temporal features to classify users as Bot or Human.
    """
    def __init__(self, temporal_in: int, spatial_in: int, hidden_dim: int = 64):
        super(HybridBotDetector, self).__init__()
        
        self.temporal_encoder = TemporalEncoder(input_dim=temporal_in, hidden_dim=hidden_dim)
        self.spatial_encoder = SpatialEncoder(in_channels=spatial_in, hidden_channels=hidden_dim, out_channels=hidden_dim * 2)
        
        # Classifier head: (Bi-GRU output = 2*hidden) + (GraphSAGE output = 2*hidden) -> 4*hidden
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, temporal_seq: torch.Tensor, node_features: torch.Tensor, edge_index: torch.Tensor, target_node_idx: int) -> torch.Tensor:
        # Extract Temporal Features
        temporal_emb = self.temporal_encoder(temporal_seq) 
        
        # Extract Spatial Features (for all nodes in the sub-graph)
        spatial_emb_all = self.spatial_encoder(node_features, edge_index)
        
        # Select only the spatial embedding for the specific user we are evaluating
        spatial_emb_target = spatial_emb_all[target_node_idx].unsqueeze(0) 
        
        # Fuse and Classify
        fused_features = torch.cat((temporal_emb, spatial_emb_target), dim=1)
        probability = self.classifier(fused_features)
        
        return probability