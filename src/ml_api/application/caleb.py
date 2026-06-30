import torch
import torch.nn as nn

class CALEBGenerator(nn.Module):
    """
    Generative network (CGAN) synthesizing evolved bot behaviors 
    to bypass spatial-temporal classifiers.
    """
    def __init__(self, latent_dim: int = 10, feature_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 16),
            nn.ReLU(),
            nn.Linear(16, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, labels], dim=1)
        return self.net(x)

class CALEBAugmenter:
    """
    Generates synthetic evasive bot samples using the trained CALEB generator.
    """
    def __init__(self, latent_dim: int = 10):
        self.latent_dim = latent_dim
        self.generator = CALEBGenerator(latent_dim=latent_dim)

    def generate_evasive_bots(self, num_samples: int = 50) -> list:
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim)
            labels = torch.ones(num_samples, 1)  # Condition on Bot target label
            synthetic_features = self.generator(noise, labels).tolist()
        
        records = []
        for i, feat in enumerate(synthetic_features):
            records.append({
                "user_id": f"caleb-synthetic-bot-{i:03d}",
                "reputation": float(feat[0] * 0.4 + 0.1),       # Mutated to look somewhat balanced
                "normalized_entropy": float(feat[1] * 0.6 + 0.3), # Increased entropy
                "spam_density": float(feat[2] * 0.4),            # Diluted spam link ratio
                "observed_label": 1,
                "true_label": 1
            })
        return records

    def generate_evasive_samples_for_training(self, num_samples: int = 20, base_template: dict = None) -> list:
        import math
        
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim)
            labels = torch.ones(num_samples, 1)  # Condition on Bot target label
            synthetic_features = self.generator(noise, labels).tolist()
            
        samples = []
        for i, feat in enumerate(synthetic_features):
            # Map CGAN outputs to feature spaces
            rep = float(feat[0] * 0.4 + 0.1)
            entropy = float(feat[1] * 0.6 + 0.3)
            spam_density = float(feat[2] * 0.4)
            avg_length = float(feat[3] * 0.8 + 0.1) if len(feat) > 3 else 0.4
            
            # Determine base graph structure
            if base_template:
                temporal = [row.copy() for row in base_template["temporal"]]
                nodes = [row.copy() for row in base_template["nodes"]]
                edges = [row.copy() for row in base_template["edges"]]
            else:
                # Default 2-node graph template
                temporal = [[0.0, 0.0] for _ in range(10)]
                nodes = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                edges = [[0, 1], [1, 0]]
                
            # 1. Perturb temporal sequence
            # Set is_complex based on spam density
            num_complex = min(max(int(round(len(temporal) * spam_density)), 0), len(temporal))
            for t_idx in range(len(temporal)):
                # Apply average length and add fluctuations based on entropy
                length_val = avg_length + 0.05 * entropy * math.sin(t_idx)
                temporal[t_idx][0] = min(max(length_val, 0.0), 1.0)
                
                # Set complexity flag
                temporal[t_idx][1] = 1.0 if t_idx < num_complex else 0.0
                
            # 2. Perturb spatial features of the target node (index 0)
            followers = 10.0
            friends = followers * (1.0 - rep) / max(rep, 0.01)
            log_followers = math.log1p(followers)
            log_friends = math.log1p(friends)
            log_ratio = math.log1p(followers / (friends + 1.0))
            
            nodes[0] = [log_followers, log_friends, log_ratio]
            
            samples.append({
                "label": 1,  # Evasive Bot
                "temporal": temporal,
                "nodes": nodes,
                "edges": edges
            })
            
        return samples
