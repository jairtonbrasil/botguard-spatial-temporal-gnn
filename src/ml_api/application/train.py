import os
import json
import csv
import logging
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from ml_api.domain.networks import HybridBotDetector
from ml_api.application.caleb import CALEBAugmenter

logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(self, weights_path: str = "data/weights/twibot_baseline.pt"):
        self.weights_path = Path(weights_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = HybridBotDetector(temporal_in=2, spatial_in=3, hidden_dim=64).to(self.device)
        if self.weights_path.exists():
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))

    def load_human_labels(self) -> list:
        csv_path = Path("data/processed/manual_labels.csv")
        if not csv_path.exists():
            return []
        
        samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    feat = json.loads(row["features"])
                    samples.append({
                        "label": int(row["labeled_as"]),
                        "temporal": feat[0],
                        "nodes": feat[1],
                        "edges": feat[2]
                    })
                except Exception:
                    continue
        return samples

    def run(self):
        logger.info("Starting model retraining flow...")
        
        human_samples = self.load_human_labels()
        logger.info(f"Loaded {len(human_samples)} expert human annotations.")
        
        augmenter = CALEBAugmenter()
        # Generate structural synthetic samples compatible with HybridBotDetector
        base_template = human_samples[0] if human_samples else None
        synthetic_samples = augmenter.generate_evasive_samples_for_training(
            num_samples=20,
            base_template=base_template
        )
        logger.info(f"Generated {len(synthetic_samples)} structural evasive bot samples via CALEB CGAN.")
        
        # Combine human annotations and synthetic CGAN samples
        training_samples = human_samples + synthetic_samples
        
        if training_samples:
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=0.005)
            criterion = nn.BCELoss()
            
            for epoch in range(5):
                total_loss = 0.0
                for sample in training_samples:
                    optimizer.zero_grad()
                    
                    temp_seq = torch.tensor([sample["temporal"]], dtype=torch.float32).to(self.device)
                    x = torch.tensor(sample["nodes"], dtype=torch.float32).to(self.device)
                    edges = torch.tensor(sample["edges"], dtype=torch.long).to(self.device)
                    target_label = torch.tensor([[float(sample["label"])]], dtype=torch.float32).to(self.device)
                    
                    pred = self.model(temp_seq, x, edges, target_node_idx=0)
                    loss = criterion(pred, target_label)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                logger.info(f"Epoch {epoch+1}/5 - Loss: {total_loss / max(len(training_samples), 1):.4f}")
        
        # Save updated weights
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.weights_path)
        logger.info(f"Retrained weights saved successfully to {self.weights_path}")
        
        # Trigger Zero-Downtime Hot-Swap
        try:
            req = urllib.request.Request(
                "http://localhost:8000/reload",
                data=b"",
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode())
                logger.info(f"Hot-Swap Status: {res_data.get('message')}")
        except Exception as e:
            logger.warning(f"ML API reload trigger skipped: {e} (Make sure the API server is running on port 8000)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    retrainer = ModelRetrainer()
    retrainer.run()
