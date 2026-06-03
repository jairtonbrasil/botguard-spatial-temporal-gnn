import json
import logging
from typing import Dict, Any
from confluent_kafka import Producer

logger = logging.getLogger(__name__)

class EventProducer:
    def __init__(self, broker_url: str = "localhost:9092", topic: str = "user_actions"):
        self.topic = topic
        self.producer = Producer({
            'bootstrap.servers': broker_url,
            'client.id': 'botguard-simulator',
            'enable.idempotence': True,
            'acks': 'all'
        })

    def _delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    def publish_action(self, action_dict: Dict[str, Any]):
        try:
            payload = json.dumps(action_dict).encode('utf-8')
            
            # Using user_id as key guarantees ordered partitions for the same user
            self.producer.produce(
                topic=self.topic,
                key=action_dict.get('user_id', '').encode('utf-8'),
                value=payload,
                callback=self._delivery_report
            )
            self.producer.poll(0)
        except Exception as e:
            logger.error(f"Failed to publish to Kafka: {e}")

    def flush(self):
        logger.info("Flushing Kafka producer...")
        self.producer.flush()