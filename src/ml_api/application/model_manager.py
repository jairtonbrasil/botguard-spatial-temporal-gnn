import logging
import torch
import os
import threading
from typing import Optional
from ml_api.domain.networks import HybridBotDetector

logger = logging.getLogger(__name__)

class InferenceManager:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing ML Inference on device: {self.device}")
        
        self.model = HybridBotDetector(temporal_in=2, spatial_in=3, hidden_dim=64).to(self.device)
        self._load_weights(model_path)
        self.model.eval()
        
        # Shadow deployment attributes
        self.shadow_model = None
        self._stats_lock = threading.Lock()
        self.shadow_stats = {
            "count": 0,
            "mae_sum": 0.0,
            "divergence_count": 0
        }
        
        shadow_path = "data/weights/twibot_shadow.pt"
        if os.path.exists(shadow_path):
            self.load_shadow(shadow_path)

    def reload(self, model_path: str):
        self._load_weights(model_path)
        self.model.eval()

    def load_shadow(self, model_path: str):
        try:
            self.shadow_model = HybridBotDetector(temporal_in=2, spatial_in=3, hidden_dim=64).to(self.device)
            self.shadow_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.shadow_model.eval()
            with self._stats_lock:
                self.shadow_stats = {
                    "count": 0,
                    "mae_sum": 0.0,
                    "divergence_count": 0
                }
            logger.info(f"Successfully loaded shadow model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load shadow weights: {e}")
            self.shadow_model = None

    def promote_shadow(self, target_path: str = "data/weights/twibot_baseline.pt"):
        if not self.shadow_model:
            raise ValueError("No shadow model loaded to promote.")
        torch.save(self.shadow_model.state_dict(), target_path)
        self.model.load_state_dict(self.shadow_model.state_dict())
        self.model.eval()
        self.shadow_model = None
        logger.info("Shadow model successfully promoted to production.")

    def _load_weights(self, model_path: Optional[str]):
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Successfully loaded model weights from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load weights from {model_path}: {e}. Using random initialization.")
        else:
            logger.warning("No model path provided or file missing. Initializing with random weights.")

    def predict(self, temporal_features: list, node_features: list, edge_index: list, target_idx: int) -> float:
        try:
            temp_seq = torch.tensor([temporal_features], dtype=torch.float32).to(self.device)
            x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
            edges = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                probability = self.model(temp_seq, x, edges, target_idx)
            return probability.item()
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return 0.0

    def predict_shadow(self, temporal_features: list, node_features: list, edge_index: list, target_idx: int) -> Optional[float]:
        if not self.shadow_model:
            return None
        try:
            temp_seq = torch.tensor([temporal_features], dtype=torch.float32).to(self.device)
            x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
            edges = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                probability = self.shadow_model(temp_seq, x, edges, target_idx)
            return probability.item()
        except Exception as e:
            logger.error(f"Shadow inference failed: {e}")
            return None

    def update_shadow_stats(self, base_score: float, shadow_score: float, base_action: str, shadow_action: str):
        with self._stats_lock:
            self.shadow_stats["count"] += 1
            self.shadow_stats["mae_sum"] += abs(base_score - shadow_score)
            if base_action != shadow_action:
                self.shadow_stats["divergence_count"] += 1

    def get_shadow_report(self) -> dict:
        with self._stats_lock:
            count = self.shadow_stats["count"]
            if count == 0:
                return {"total_comparisons": 0, "mean_absolute_error": 0.0, "divergence_rate": 0.0, "status": "idle"}
            return {
                "total_comparisons": count,
                "mean_absolute_error": round(self.shadow_stats["mae_sum"] / count, 6),
                "divergence_rate": round(self.shadow_stats["divergence_count"] / count, 4),
                "status": "active"
            }

    def explain(self, temporal_features: list, node_features: list, edge_index: list, target_idx: int) -> dict:
        try:
            n_nodes = len(node_features)
            
            # Prepare the 6 temporal sequence variants
            temp_seq_0 = temporal_features
            temp_seq_1 = temporal_features
            temp_seq_2 = temporal_features
            temp_seq_3 = temporal_features
            temp_seq_4 = [[0.0, row[1]] for row in temporal_features]
            temp_seq_5 = [[row[0], 0.0] for row in temporal_features]
            
            temp_seq_batch = torch.tensor(
                [temp_seq_0, temp_seq_1, temp_seq_2, temp_seq_3, temp_seq_4, temp_seq_5],
                dtype=torch.float32
            ).to(self.device)
            
            # Prepare the 6 node features variants
            nf_0 = node_features
            
            nf_1 = [row.copy() for row in node_features]
            nf_1[target_idx][0] = 0.0
            
            nf_2 = [row.copy() for row in node_features]
            nf_2[target_idx][1] = 0.0
            
            nf_3 = [row.copy() for row in node_features]
            nf_3[target_idx][2] = 0.0
            
            nf_4 = node_features
            nf_5 = node_features
            
            all_node_features = []
            for nf in [nf_0, nf_1, nf_2, nf_3, nf_4, nf_5]:
                all_node_features.extend(nf)
            node_features_batch = torch.tensor(all_node_features, dtype=torch.float32).to(self.device)
            
            # Prepare the 6 edge indices variants (shifted)
            edges_tensor = torch.tensor(edge_index, dtype=torch.long)
            edge_index_list = []
            for i in range(6):
                edge_index_list.append(edges_tensor + i * n_nodes)
            edge_index_batch = torch.cat(edge_index_list, dim=1).to(self.device)
            
            # Target indices for the batch
            target_idxs_batch = torch.tensor([target_idx + i * n_nodes for i in range(6)], dtype=torch.long).to(self.device)
            
            # Single parallel batch inference
            with torch.no_grad():
                probabilities = self.model(temp_seq_batch, node_features_batch, edge_index_batch, target_idxs_batch)
                
            scores = probabilities.squeeze(1).tolist()
            base_score = scores[0]
            score_followers = scores[1]
            score_following = scores[2]
            score_ratio = scores[3]
            score_len = scores[4]
            score_complex = scores[5]
            
            d_followers = abs(base_score - score_followers)
            d_following = abs(base_score - score_following)
            d_ratio = abs(base_score - score_ratio)
            d_len = abs(base_score - score_len)
            d_complex = abs(base_score - score_complex)
            
            total = d_followers + d_following + d_ratio + d_len + d_complex
            if total > 0.0:
                return {
                    "followers_count": round(d_followers / total, 4),
                    "following_count": round(d_following / total, 4),
                    "follower_ratio": round(d_ratio / total, 4),
                    "post_length": round(d_len / total, 4),
                    "post_complexity": round(d_complex / total, 4)
                }
        except Exception as e:
            logger.error(f"Explainability batch inference failed: {e}")
            
        return {
            "followers_count": 0.20,
            "following_count": 0.20,
            "follower_ratio": 0.20,
            "post_length": 0.20,
            "post_complexity": 0.20
        }