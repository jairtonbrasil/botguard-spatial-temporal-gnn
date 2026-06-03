# BotGuard Machine Learning Architecture: Technical Specifications

## 1. Architectural Overview & Hybrid Processing

The BotGuard Machine Learning pipeline is designed to resolve adversarial bot traffic using a multi-modal neural network. Because social bots can mimic human language and temporal posting frequencies or manipulate network graphs locally, a single-mode classifier is easily bypassed. 

The `HybridBotDetector` solves this problem by combining spatial (topological) features from a Graph Neural Network with temporal (sequential) features from a Recurrent Neural Network:

- **Temporal Encoder (Bi-GRU)**: Extracts behavioral entropy, velocity, and action sequence patterns from the user's localized chronological timeline.
- **Spatial Encoder (GraphSAGE)**: Extracts structural interaction context, community isolation, and neighborhood asymmetry from the user's graph sub-network.

```
                  +-------------------------------------------------------------+
                  |                 Hybrid Classifier Data Flow                 |
                  +-------------------------------------------------------------+

   Temporal Action Sequence                           Graph Neighborhood Topology
  [Batch, SeqLen, InputDim]                         [Nodes, Features] & [2, Edges]
              |                                                    |
              v                                                    v
      [TemporalEncoder]                                     [SpatialEncoder]
       (Bi-directional GRU)                                 (2-layer GraphSAGE)
              |                                                    |
              | Cat hidden[-2] & hidden[-1]                        | SAGEConv Layers
              v                                                    v
      Temporal Embedding                                    Spatial Embedding
    [Batch, 2 * HiddenDim]                                [Nodes, 2 * HiddenDim]
              |                                                    |
              |                                                    | Extract evaluated
              |                                                    | target node index
              |                                                    v
              |                                             Target Embedding
              |                                           [Batch, 2 * HiddenDim]
              |                                                    |
              +-------------------------+--------------------------+
                                        |
                                        v Concatenation
                                  Fused Vector
                              [Batch, 4 * HiddenDim]
                                        |
                                        v Sequential Classifier Head
                                  Fully Connected
                                        |
                                        v Dropout (p = 0.3) & ReLU
                                  Binary Sigmoid
                                        |
                                        v
                                 P(Bot) Prediction
```

---

## 2. Deep Dive: Temporal Encoder (`TemporalEncoder`)

The `TemporalEncoder` converts chronological sliding-window event sequences into compact behavioral representations. It parses sequence patterns to identify mechanical velocities or automated, repeating transaction loops.

```python
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(TemporalEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
```

### 2.1 Bidirectional Processing (`bidirectional=True`)
Traditional sequential models process data in a single direction (past-to-present). However, in modern social bot detection, the full context of a timeline is crucial across both directions. 

- **Context Hijacking**: Advanced bots frequently insert organic-looking posts (e.g., standard comments about the weather or commute) as a smoke screen before initiating a high-velocity malicious giveaway campaign. 
- **Backward Parsing**: Running a bidirectional Gated Recurrent Unit (GRU) cell allows the network to process the timeline forward (capturing standard state transitions) and backward (evaluating past events conditioned on subsequent anomalies). This enables the network to identify that a seemingly organic post was actually a preparation phase for automated spam.

The Gated Recurrent Unit is formulated using standard update and reset gates at time step $t$:
- **Reset Gate**: $r_t = \sigma(W_r x_t + U_r h_{t-1})$
- **Update Gate**: $z_t = \sigma(W_z x_t + U_z h_{t-1})$
- **Candidate Hidden State**: $\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}))$
- **Final Hidden State**: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

### 2.2 Memory Alignment (`batch_first=True`)
PyTorch sequential modules default to receiving tensor dimensions formatted as `[SequenceLength, BatchSize, InputDim]`. 

To support real-time sub-millisecond API inference, the network overrides this layout setting `batch_first=True`, alining the expected tensor input shape to:
$$\text{Shape}(x) = \left[ \text{BatchSize}, \text{SequenceLength}, \text{InputDim} \right]$$
This matches the exact structure of deserialized sliding windows retrieved from Redis. Bypassing expensive transposition or memory-copy operations during inference maximizes throughput in high-velocity stream processing.

### 2.3 Dual-Directional Feature Fusion
Because the GRU is bidirectional, it produces two independent sequences of hidden states. The final step of the encoder fuses these states into a unified temporal summary vector:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    output, hidden = self.gru(x)
    # Concatenate the final hidden states from both directions
    forward_hidden = hidden[-2, :, :]
    backward_hidden = hidden[-1, :, :]
    return torch.cat((forward_hidden, backward_hidden), dim=1)
```

- **Hidden Indexing**: The final hidden state matrix contains tensors corresponding to all layers and directions. Since we employ a single-layer GRU, `hidden[-2]` extracts the last hidden state vector of the forward pass ($\vec{h}_T$), and `hidden[-1]` extracts the last hidden state vector of the backward pass ($\overleftarrow{h}_1$).
- **Concatenation Fusion**: Concatenating both vectors results in a dense behavioral profile representation:
  $$h_{\text{temporal}} = \text{CONCAT} \left( \vec{h}_T, \overleftarrow{h}_1 \right) \in \mathbb{R}^{\text{BatchSize} \times (2 \times \text{HiddenDim})}$$

---

## 3. Deep Dive: Spatial Encoder (`SpatialEncoder`)

The `SpatialEncoder` maps the relational topological features surrounding a user node. It evaluates social connectivity properties to isolate automated coordinate groups.

```python
class SpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(SpatialEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
```

### 3.1 k-Hop Neighborhood Context
In a Graph Neural Network (GNN), the depth of the layers corresponds directly to the size of the neighborhood neighborhood context (the k-hop distance) aggregated around the target vertex.

- **Layer 1 (`conv1`)**: Aggregates properties from 1-hop direct neighbors (immediate followers and followed users).
- **Layer 2 (`conv2`)**: Aggregates properties from 2-hop neighbors (neighbors of neighbors).
- **Scientific Standard**: In social graphs, bot coordinators can easily construct artificial 1-hop interactions (e.g., buying fake followers or forming mutual follow loops). However, infiltrating 2-hop structures organically is extremely difficult. The 2-hop scope serves as the scientific baseline for isolating automated accounts.
- **Over-Smoothing Prevention**: Restricting GNN aggregation to exactly 2 layers prevents the "over-smoothing" problem, where adding too many layers causes node embeddings across the entire graph to converge and become indistinguishable.

The message aggregation formula for SAGEConv at layer $k$ is given by:
$$h_{v}^{k} = W^k_{\text{self}} \cdot h_{v}^{k-1} + W^k_{\text{neigh}} \cdot \text{AGG} \left( \{ h_{u}^{k-1}, \forall u \in \mathcal{N}(v) \} \right)$$

### 3.2 Feature Weight Balancing (`out_channels = hidden_dim * 2`)
To avoid structural bias during fusion, spatial features must not overpower temporal features (and vice versa).

The `SpatialEncoder` outputs a projection layer of dimension `hidden_dim * 2` (128 dimensions for `hidden_dim = 64`). This matches the exact size of the concatenated bidirectional temporal state:
$$\text{Dim}(h_{\text{spatial}}) = \text{Dim}(h_{\text{temporal}}) = 2 \times \text{HiddenDim} = 128$$
This balanced dimensions constraint ensures that spatial graph features and temporal timeline sequences exert equal gradients during model training.

---

## 4. Deep Dive: Hybrid Detector (`HybridBotDetector`)

The `HybridBotDetector` orchestrates the complete inference lifecycle. It extracts features from both models, fuses them into a combined representation, and maps it to a binary classification probability.

```python
class HybridBotDetector(nn.Module):
    def __init__(self, temporal_in: int, spatial_in: int, hidden_dim: int = 64):
        super(HybridBotDetector, self).__init__()
        self.temporal_encoder = TemporalEncoder(input_dim=temporal_in, hidden_dim=hidden_dim)
        self.spatial_encoder = SpatialEncoder(in_channels=spatial_in, hidden_channels=hidden_dim, out_channels=hidden_dim * 2)
```

### 4.1 Empirical Dimensioning (`hidden_dim = 64`)
The baseline capacity is pinned to a standard state of `hidden_dim = 64`. This value represents a balanced empirical configuration:
- **Low Capacity (e.g., 16)**: Prevents the network from mapping complex behavioral features, resulting in high underfitting on advanced evasion strategies.
- **High Capacity (e.g., 256 or 512)**: Drastically increases system latency during real-time streaming ingestion and causes rapid model overfitting on synthetic datasets.

### 4.2 Dimensional Fusion Math
When concatenating the two state representations, the input dimension for the classification head must be defined explicitly:

```python
self.classifier = nn.Sequential(
    nn.Linear(hidden_dim * 4, hidden_dim),
    nn.ReLU(),
    # ...
)
```

The output sizes of both modules dictate the input layer dimension of `hidden_dim * 4` (256):
- **Temporal Profile Vector ($\text{Dim} = 2 \times 64 = 128$)**: Bi-directional concatenated states of the GRU cell.
- **Spatial Topology Vector ($\text{Dim} = 2 \times 64 = 128$)**: Output layer representation of the GraphSAGE network.
- **Total Linear Input**:
  $$\text{InputDim} = \text{Dim}(h_{\text{temporal}}) + \text{Dim}(h_{\text{spatial}}) = 128 + 128 = 256 = 4 \times \text{HiddenDim}$$

### 4.3 Generalization Regularization (`Dropout(p=0.3)`)
To avoid memorization on simulated datasets, the model applies a dropout rate of $30\%$ within the convolutional layers and classification head:
- **Overfitting Risk**: Bots generated in simulated environments often follow repetitive structural patterns. Without regularization, deep classifiers will simply memorize these static graph structures (overfitting subgraph configurations).
- **Forced Adaptation**: By randomly dropping $30\%$ of activation pathways during training epochs, the network is forced to learn robust generalized behavioral signatures (entropy values, dynamic follower ratios) rather than relying on specific static nodes.

---

## 5. Forward Execution Logic

During inference, the model processes temporal histories and graph states concurrently, extracting target vectors dynamically:

```python
def forward(self, temporal_seq: torch.Tensor, node_features: torch.Tensor, edge_index: torch.Tensor, target_node_idx: int) -> torch.Tensor:
    # 1. Extract Temporal Features
    temporal_emb = self.temporal_encoder(temporal_seq) 
    
    # 2. Extract Spatial Features (for all nodes in the sub-graph)
    spatial_emb_all = self.spatial_encoder(node_features, edge_index)
    
    # Select only the spatial embedding for the specific user we are evaluating
    spatial_emb_target = spatial_emb_all[target_node_idx].unsqueeze(0) 
    
    # 3. Fuse and Classify
    fused_features = torch.cat((temporal_emb, spatial_emb_target), dim=1)
    probability = self.classifier(fused_features)
    
    return probability
```

1. **Temporal Extraction**: The sequence `temporal_seq` is encoded to yield a timeline behavior embedding of shape `[1, 128]`.
2. **Spatial Graph Aggregation**: GraphSAGE computes structural representation embeddings for *all* vertices in the input subgraph (`spatial_emb_all`).
3. **Target Node Slicing**: The network slices the target node representation (`target_node_idx`) and unsqueezes it, resulting in a vector of shape `[1, 128]`.
4. **Feature Fusion**: The vectors are joined along the feature axis (`dim=1`), creating a combined feature vector of shape `[1, 256]`.
5. **Probability Output**: The classifier projects the combined vector, applying ReLU and Sigmoid activation layers to output $P(\text{Bot}) \in [0, 1]$.
