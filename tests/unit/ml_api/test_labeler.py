import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime, timezone

from ml_api.application.labeler_heuristico import CresciHeuristicLabeler

class TestCresciHeuristicLabeler(unittest.TestCase):
    @patch('ml_api.application.labeler_heuristico.GraphStoreClient')
    @patch('ml_api.application.labeler_heuristico.TimeSeriesClient')
    def setUp(self, mock_redis_client, mock_neo4j_client):
        # Create our labeler with mocked internal clients
        self.labeler = CresciHeuristicLabeler()
        
    def test_calculate_reputation_human(self):
        # Setup mock Neo4j record for a normal human: 150 followers, 100 following, true_label = 0
        mock_session = MagicMock()
        self.labeler.graph_store.driver.session.return_value.__enter__.return_value = mock_session  # type: ignore
        
        mock_result = MagicMock()
        mock_record = {
            "followers": 150,
            "following": 100,
            "true_label": 0
        }
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        
        reputation, followers, following, true_label = self.labeler.calculate_reputation("human_user_id")
        
        self.assertEqual(followers, 150)
        self.assertEqual(following, 100)
        self.assertEqual(true_label, 0)
        # R = 150 / 250 = 0.60
        self.assertAlmostEqual(reputation, 0.60)

    def test_calculate_reputation_bot(self):
        # Setup mock Neo4j record for an aggressive spam bot: 5 followers, 495 following, true_label = 1
        mock_session = MagicMock()
        self.labeler.graph_store.driver.session.return_value.__enter__.return_value = mock_session  # type: ignore
        
        mock_result = MagicMock()
        mock_record = {
            "followers": 5,
            "following": 495,
            "true_label": 1
        }
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        
        reputation, followers, following, true_label = self.labeler.calculate_reputation("bot_user_id")
        
        self.assertEqual(followers, 5)
        self.assertEqual(following, 495)
        self.assertEqual(true_label, 1)
        # R = 5 / 500 = 0.01
        self.assertAlmostEqual(reputation, 0.01)

    def test_calculate_entropy_human_organic(self):
        # Set up a mock human Redis timeline with varied time gaps and no spam elements (is_complex = 0.0)
        # Chronological order from oldest to newest:
        # t0: 10:00:00, t1: 10:00:05 (delta = 5s -> B1)
        # t2: 10:05:00 (delta = 295s -> B2)
        # t3: 12:00:00 (delta = 6900s -> B3)
        # t4: 2026-06-03 (delta > 86400s -> B4)
        mock_records = [
            json.dumps({"ts": "2026-06-02T10:00:00Z", "is_complex": 0.0}),
            json.dumps({"ts": "2026-06-02T10:00:05Z", "is_complex": 0.0}),
            json.dumps({"ts": "2026-06-02T10:05:00Z", "is_complex": 0.0}),
            json.dumps({"ts": "2026-06-02T12:00:00Z", "is_complex": 0.0}),
            json.dumps({"ts": "2026-06-03T15:00:00Z", "is_complex": 0.0})
        ]
        # Redis returns newest first, so reverse order
        mock_records.reverse()
        self.labeler.time_series.redis.lrange.return_value = mock_records
        
        entropy, normalized_entropy, spam_density = self.labeler.calculate_temporal_entropy("human_user_id")
        
        # Timeline had 0 complex actions, spam density should be 0.0
        self.assertEqual(spam_density, 0.0)
        # Time intervals are spread across multiple buckets (B1, B2, B3, B4), so entropy must be > 0.0
        self.assertGreater(entropy, 1.0)
        self.assertGreater(normalized_entropy, 0.50)

    def test_calculate_entropy_bot_constant_intervals(self):
        # Setup a mock bot timeline posting exactly every 30 seconds (delta = 30s -> all B1)
        # with 100% spam density (is_complex = 1.0)
        mock_records = [
            json.dumps({"ts": "2026-06-02T10:00:00Z", "is_complex": 1.0}),
            json.dumps({"ts": "2026-06-02T10:00:30Z", "is_complex": 1.0}),
            json.dumps({"ts": "2026-06-02T10:01:00Z", "is_complex": 1.0}),
            json.dumps({"ts": "2026-06-02T10:01:30Z", "is_complex": 1.0})
        ]
        mock_records.reverse()
        self.labeler.time_series.redis.lrange.return_value = mock_records
        
        entropy, normalized_entropy, spam_density = self.labeler.calculate_temporal_entropy("bot_user_id")
        
        # 100% complex actions, spam density should be 1.0
        self.assertEqual(spam_density, 1.0)
        # Because all deltas are exactly 30s, they fall into a single bucket (B1), so entropy is exactly 0.0
        self.assertAlmostEqual(entropy, 0.0)
        self.assertAlmostEqual(normalized_entropy, 0.0)

    @patch('ml_api.application.labeler_heuristico.CresciHeuristicLabeler.calculate_reputation')
    @patch('ml_api.application.labeler_heuristico.CresciHeuristicLabeler.calculate_temporal_entropy')
    def test_evaluate_user_human(self, mock_entropy, mock_reputation):
        # Force metric outputs for a clear human profile
        mock_reputation.return_value = (0.75, 150, 50, 0) # Reputation = 0.75
        mock_entropy.return_value = (1.8, 0.80, 0.10)     # Normalized Entropy = 0.80, Spam Density = 0.10
        
        result = self.labeler.evaluate_user("human_user_id")
        
        # Score = 0.35 * (1 - 0.75) + 0.35 * (1 - 0.80) + 0.30 * 0.10
        # Score = 0.0875 + 0.07 + 0.03 = 0.1875
        self.assertAlmostEqual(result["heuristic_score"], 0.1875)
        self.assertEqual(result["observed_label"], 0) # Cleared as Human

    @patch('ml_api.application.labeler_heuristico.CresciHeuristicLabeler.calculate_reputation')
    @patch('ml_api.application.labeler_heuristico.CresciHeuristicLabeler.calculate_temporal_entropy')
    def test_evaluate_user_bot(self, mock_entropy, mock_reputation):
        # Force metric outputs for a clear bot profile
        mock_reputation.return_value = (0.02, 10, 490, 1) # Reputation = 0.02
        mock_entropy.return_value = (0.0, 0.0, 0.95)      # Normalized Entropy = 0.0, Spam Density = 0.95
        
        result = self.labeler.evaluate_user("bot_user_id")
        
        # Score = 0.35 * (1 - 0.02) + 0.35 * (1 - 0.0) + 0.30 * 0.95
        # Score = 0.343 + 0.35 + 0.285 = 0.978
        self.assertAlmostEqual(result["heuristic_score"], 0.978)
        self.assertEqual(result["observed_label"], 1) # Banned/Observed as Bot

if __name__ == "__main__":
    unittest.main()
