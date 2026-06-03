import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import json
import logging
from confluent_kafka import Consumer, KafkaError, Producer
from stream_processor.infrastructure.neo4j_client import GraphStoreClient
from stream_processor.infrastructure.redis_client import TimeSeriesClient
from stream_processor.infrastructure.ml_client import MachineLearningClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class StreamProcessorOrchestrator:
    def __init__(self, broker_url: str = "localhost:9092", topic: str = "user_actions"):
        self.consumer = Consumer({
            'bootstrap.servers': broker_url,
            'group.id': 'botguard-state-processor',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        })
        self.topic = topic
        self.consumer.subscribe([self.topic])
        
        self.producer = Producer({
            'bootstrap.servers': broker_url
        })
        self.dlq_topic = "user_actions_dlq"
        
        self.graph_store = GraphStoreClient()
        self.time_series = TimeSeriesClient()
        self.ml_client = MachineLearningClient()

    def _extract_features(self, user_id: str) -> dict:
        temporal_seq = self.time_series.get_timeline_features(user_id, limit=10)
        graph_data = self.graph_store.get_subgraph_features(user_id)
        return {
            "user_id": user_id,
            "target_node_idx": graph_data["target_node_idx"],
            "temporal_features": temporal_seq,
            "node_features": graph_data["node_features"],
            "edge_index": graph_data["edge_index"]
        }

    def _send_to_dlq(self, raw_value: bytes, error_message: str):
        try:
            try:
                decoded_val = raw_value.decode('utf-8')
            except Exception:
                decoded_val = str(raw_value)
            
            dlq_payload = {
                "raw_message": decoded_val,
                "error": error_message
            }
            self.producer.produce(
                self.dlq_topic,
                value=json.dumps(dlq_payload).encode('utf-8')
            )
            self.producer.flush(0.5)
            logger.warning(f"Sent message to DLQ: {error_message}")
        except Exception as pe:
            logger.error(f"Failed to publish to DLQ: {pe}")

    def run_continuously(self):
        logger.info("Starting Stream Processor with ML Inference integration...")
        try:
            while True:
                msg = self.consumer.poll(1.0)

                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka consumer error: {msg.error()}")
                        break

                try:
                    payload = json.loads(msg.value().decode('utf-8'))
                    user_id = payload.get('user_id')
                    if not user_id:
                        raise ValueError("Missing user_id in payload")
                    
                    self.graph_store.update_topology(payload)
                    self.time_series.record_action(payload)
                    
                    features = self._extract_features(user_id)
                    decision = self.ml_client.evaluate_user(features)
                    
                    if decision is None:
                        raise Exception("ML API communication timeout/failure")
                    
                    prob = decision.get("bot_probability", 0)
                    action = decision.get("action")
                    needs_review = decision.get("needs_manual_review")
                    
                    if action == "BAN":
                        logger.warning(f"🚫 [BAN] User {user_id[:8]} blocked! (P = {prob:.4f})")
                    elif action == "LIMIT":
                        logger.warning(f"⚠️ [LIMIT] User {user_id[:8]} rate-limited! (P = {prob:.4f})")
                    else:
                        logger.info(f"✅ [ALLOW] User {user_id[:8]} cleared. (P = {prob:.4f})")
                        
                    if needs_review:
                        review_payload = {
                            "user_id": user_id,
                            "bot_probability": prob,
                            "timestamp": payload.get("timestamp"),
                            "content": payload.get("content") or "",
                            "features": features
                        }
                        try:
                            self.time_series.redis.lpush("active_learning:queue", json.dumps(review_payload))
                            self.time_series.redis.ltrim("active_learning:queue", 0, 199)
                        except Exception as re:
                            logger.error(f"Failed to queue active learning event: {re}")
                    
                    self.consumer.commit(asynchronous=True)
                    
                except json.JSONDecodeError as jde:
                    logger.error("Failed to decode message payload.")
                    self._send_to_dlq(msg.value(), f"JSONDecodeError: {jde}")
                    self.consumer.commit(asynchronous=True)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self._send_to_dlq(msg.value(), f"ProcessingError: {e}")
                    self.consumer.commit(asynchronous=True)

        except KeyboardInterrupt:
            logger.info("Stream Processor stopped.")
        finally:
            self.consumer.close()
            self.graph_store.close()

if __name__ == "__main__":
    processor = StreamProcessorOrchestrator()
    processor.run_continuously()