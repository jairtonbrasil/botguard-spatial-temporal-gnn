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
