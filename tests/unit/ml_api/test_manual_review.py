import unittest
from unittest.mock import MagicMock, patch, mock_open
import json
from pathlib import Path

from ml_api.application.manual_review import ActiveLearningReviewer

class TestActiveLearningReviewer(unittest.TestCase):
    @patch('ml_api.application.manual_review.redis.Redis')
    def setUp(self, mock_redis_class):
        self.reviewer = ActiveLearningReviewer()
        self.reviewer.output_file = Path("dummy_manual_labels.csv")

    def test_load_queue_empty(self):
        self.reviewer.redis.lrange.return_value = []  # type: ignore
        result = self.reviewer.load_queue()
        self.assertEqual(result, [])

    def test_load_queue_with_items(self):
        mock_payload = {
            "user_id": "test-user-123",
            "bot_probability": 0.52,
            "timestamp": "2026-06-02T01:00:00Z",
            "content": "This is an uncertain post.",
            "features": {}
        }
        self.reviewer.redis.lrange.return_value = [json.dumps(mock_payload)]  # type: ignore
        
        result = self.reviewer.load_queue()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["user_id"], "test-user-123")
        self.assertEqual(result[0]["bot_probability"], 0.52)

    @patch('builtins.open', new_callable=mock_open)
    def test_save_label_bot(self, mock_file):
        event = {
            "user_id": "user-456",
            "bot_probability": 0.48,
            "content": "Spam crypto!",
            "features": {"node_features": [[1, 2]]}
        }
        self.reviewer.save_label(event, 1)
        
        mock_file.assert_called_once_with(self.reviewer.output_file, "a", newline="", encoding="utf-8")
        
        # Verify that CSV write calls occurred
        handle = mock_file()
        handle.write.assert_called()

    def test_remove_from_queue(self):
        event_str = json.dumps({"user_id": "user-123"})
        self.reviewer.remove_from_queue(event_str)
        self.reviewer.redis.lrem.assert_called_once_with("active_learning:queue", 1, event_str)  # type: ignore

if __name__ == "__main__":
    unittest.main()
