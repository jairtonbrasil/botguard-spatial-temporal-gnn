import unittest
from unittest.mock import MagicMock, patch
import torch
import os
from pathlib import Path

from ml_api.application.model_manager import InferenceManager

class TestMarco6Inference(unittest.TestCase):
    def setUp(self):
        # Initialize InferenceManager with random weights (bypassing file existence check)
        self.manager = InferenceManager()

    def test_explainability_contributions(self):
        temporal_features = [[0.1, 0.2] for _ in range(10)]
        node_features = [
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 2.0]
        ]
        edge_index = [[0, 1, 0, 1], [1, 0, 0, 1]]
        
        explanation = self.manager.explain(
            temporal_features=temporal_features,
            node_features=node_features,
            edge_index=edge_index,
            target_idx=0
        )
        
        # Verify all 5 features are explained
        expected_keys = {"followers_count", "following_count", "follower_ratio", "post_length", "post_complexity"}
        self.assertEqual(set(explanation.keys()), expected_keys)
        
        # Verify that contributions sum to ~1.0
        total_sum = sum(explanation.values())
        self.assertAlmostEqual(total_sum, 1.0, places=2)

    def test_shadow_mode_lifecycle(self):
        # 1. Create a dummy model file for shadow loading
        dummy_path = "data/weights/twibot_test_shadow.pt"
        os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
        torch.save(self.manager.model.state_dict(), dummy_path)
        
        try:
            # 2. Load shadow weights
            self.manager.load_shadow(dummy_path)
            self.assertIsNotNone(self.manager.shadow_model)
            
            # 3. Predict shadow
            temporal_features = [[0.1, 0.2] for _ in range(10)]
            node_features = [[1.0, 2.0, 0.5]]
            edge_index = [[0], [0]]
            
            shadow_score = self.manager.predict_shadow(temporal_features, node_features, edge_index, 0)
            self.assertIsNotNone(shadow_score)
            self.assertTrue(0.0 <= shadow_score <= 1.0)
            
            # 4. Update stats
            self.manager.update_shadow_stats(0.5, 0.6, "ALLOW", "ALLOW")
            self.manager.update_shadow_stats(0.8, 0.95, "LIMIT", "BAN")
            
            report = self.manager.get_shadow_report()
            self.assertEqual(report["total_comparisons"], 2)
            self.assertAlmostEqual(report["mean_absolute_error"], 0.125, places=4)
            self.assertEqual(report["divergence_rate"], 0.5)
            self.assertEqual(report["status"], "active")
            
            # 5. Promote shadow
            self.manager.promote_shadow(target_path=dummy_path)
            self.assertIsNone(self.manager.shadow_model)
            
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

if __name__ == "__main__":
    unittest.main()
