import unittest
from unittest.mock import MagicMock, patch, mock_open
import json
from pathlib import Path
import torch

from ml_api.application.caleb import CALEBAugmenter
from ml_api.application.train import ModelRetrainer

class TestRetrainingPipeline(unittest.TestCase):
    def test_caleb_generation(self):
        augmenter = CALEBAugmenter()
        samples = augmenter.generate_evasive_bots(num_samples=5)
        
        self.assertEqual(len(samples), 5)
        self.assertTrue(samples[0]["user_id"].startswith("caleb-synthetic-bot-"))
        self.assertEqual(samples[0]["observed_label"], 1)
        self.assertEqual(samples[0]["true_label"], 1)
        self.assertGreaterEqual(samples[0]["reputation"], 0.1)
        self.assertLessEqual(samples[0]["reputation"], 0.5)

    @patch('ml_api.application.train.HybridBotDetector.load_state_dict')
    @patch('ml_api.application.train.torch.load')
    @patch('ml_api.application.train.urllib.request.urlopen')
    def test_load_human_labels(self, mock_urlopen, mock_torch_load, mock_load_state_dict):
        # Mock CSV contents
        csv_data = (
            "user_id,bot_probability,labeled_as,content,features\n"
            "user-1,0.52,1,spam text,\"[ [[0.1, 0.2]], [[1.0, 2.0, 3.0]], [[0], [0]] ]\"\n"
        )
        
        # Patch exists to True for manual labels
        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_data)):
                retrainer = ModelRetrainer("dummy_weights.pt")
                samples = retrainer.load_human_labels()
                
                self.assertEqual(len(samples), 1)
                self.assertEqual(samples[0]["label"], 1)
                self.assertEqual(samples[0]["temporal"], [[0.1, 0.2]])

    @patch('ml_api.application.train.urllib.request.urlopen')
    def test_model_training_pass(self, mock_urlopen):
        # Create a mock response for the hot-swap
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"message": "reloaded"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        retrainer = ModelRetrainer("dummy_weights.pt")
        
        # Mock human samples to bypass real file system reads
        retrainer.load_human_labels = MagicMock(return_value=[{
            "label": 1,
            "temporal": [[0.1, 0.2]] * 10,
            "nodes": [[1.0, 2.0, 3.0]],
            "edges": [[0], [0]]
        }])
        
        # Mock saving the state dictionary
        with patch('torch.save') as mock_save:
            retrainer.run()
            mock_save.assert_called_once()

if __name__ == "__main__":
    unittest.main()
