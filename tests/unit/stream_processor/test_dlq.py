import unittest
from unittest.mock import MagicMock, patch
import json
from confluent_kafka import KafkaError
from stream_processor.main import StreamProcessorOrchestrator

class TestStreamProcessorDLQ(unittest.TestCase):
    @patch('stream_processor.main.Consumer')
    @patch('stream_processor.main.Producer')
    @patch('stream_processor.main.GraphStoreClient')
    @patch('stream_processor.main.TimeSeriesClient')
    @patch('stream_processor.main.MachineLearningClient')
    def setUp(self, mock_ml, mock_ts, mock_graph, mock_prod, mock_cons):
        self.orchestrator = StreamProcessorOrchestrator()
        self.orchestrator.consumer = MagicMock()
        self.orchestrator.producer = MagicMock()
        self.orchestrator.graph_store = MagicMock()
        self.orchestrator.time_series = MagicMock()
        self.orchestrator.ml_client = MagicMock()

    def test_decode_error_sends_to_dlq(self):
        # 1. Mock a broken JSON message
        mock_msg = MagicMock()
        mock_msg.error.return_value = None
        mock_msg.value.return_value = b"{broken_json"
        
        # Setup run_continuously loop to run once
        self.orchestrator.consumer.poll.side_effect = [mock_msg, KeyboardInterrupt()]
        
        try:
            self.orchestrator.run_continuously()
        except KeyboardInterrupt:
            pass

        # 2. Verify _send_to_dlq was triggered
        self.orchestrator.producer.produce.assert_called_once()
        args, kwargs = self.orchestrator.producer.produce.call_args
        self.assertEqual(args[0], "user_actions_dlq")
        
        payload = json.loads(kwargs["value"].decode('utf-8'))
        self.assertEqual(payload["raw_message"], "{broken_json")
        self.assertIn("JSONDecodeError", payload["error"])
        
        # 3. Verify offset was committed to not block partition
        self.orchestrator.consumer.commit.assert_called_once_with(asynchronous=True)

    def test_ml_timeout_sends_to_dlq(self):
        # 1. Mock a valid JSON message
        mock_payload = {"user_id": "test-user-999", "timestamp": "2026-06-02T21:00:00Z", "action_type": "POST"}
        mock_msg = MagicMock()
        mock_msg.error.return_value = None
        mock_msg.value.return_value = json.dumps(mock_payload).encode('utf-8')
        
        # Setup ML Client to return None (simulating API failure/timeout)
        self.orchestrator.ml_client.evaluate_user.return_value = None
        self.orchestrator.consumer.poll.side_effect = [mock_msg, KeyboardInterrupt()]
        
        try:
            self.orchestrator.run_continuously()
        except KeyboardInterrupt:
            pass

        # 2. Verify _send_to_dlq was triggered
        self.orchestrator.producer.produce.assert_called_once()
        args, kwargs = self.orchestrator.producer.produce.call_args
        self.assertEqual(args[0], "user_actions_dlq")
        
        payload = json.loads(kwargs["value"].decode('utf-8'))
        self.assertEqual(json.loads(payload["raw_message"])["user_id"], "test-user-999")
        self.assertIn("ML API communication timeout", payload["error"])
        
        # 3. Verify offset was committed
        self.orchestrator.consumer.commit.assert_called_once_with(asynchronous=True)

if __name__ == "__main__":
    unittest.main()
