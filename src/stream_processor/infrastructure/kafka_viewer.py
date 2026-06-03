import json
import logging
from confluent_kafka import Consumer, KafkaError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


COLOR_BOT = "\033[91m"     
COLOR_HUMAN = "\033[92m"   
COLOR_RESET = "\033[0m"
COLOR_URL = "\033[94m"     
COLOR_TAG = "\033[95m"     

def format_content(text: str) -> str:
    """Destaca URLs e hashtags para melhor visualização."""
    words = text.split()
    formatted = []
    for word in words:
        if word.startswith("http"):
            formatted.append(f"{COLOR_URL}{word}{COLOR_RESET}")
        elif word.startswith("#"):
            formatted.append(f"{COLOR_TAG}{word}{COLOR_RESET}")
        else:
            formatted.append(word)
    return " ".join(formatted)

class StreamViewer:
    def __init__(self, broker_url: str = "localhost:9092", topic: str = "user_actions"):
        self.consumer = Consumer({
            'bootstrap.servers': broker_url,
            'group.id': 'botguard-cli-viewer',
            'auto.offset.reset': 'latest' 
        })
        self.topic = topic
        self.consumer.subscribe([self.topic])

    def start_viewing(self):
        logger.info(f"Subscribed to topic '{self.topic}'. Waiting for real-time events...\n")
        logger.info("-" * 80)
        try:
            while True:
                msg = self.consumer.poll(1.0) 

                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue 
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        break
                payload = json.loads(msg.value().decode('utf-8'))
                
                user_id = payload.get("user_id", "Unknown")[:8] 
                action = payload.get("action_type")
                content = payload.get("content")
                true_label = payload.get("true_label")
                
                if true_label == 1:
                    header = f"{COLOR_BOT}[BOT DETECTADO - AÇÃO: {action}] {COLOR_RESET} Usuário: {user_id}"
                else:
                    header = f"{COLOR_HUMAN}[AÇÃO HUMANA - AÇÃO: {action}] {COLOR_RESET} Usuário: {user_id}"
                
                print(header)
                if content:
                    print(f"Texto: {format_content(content)}")
                print("-" * 80)

        except KeyboardInterrupt:
            logger.info("\nViewer stopped by user.")
        finally:
            self.consumer.close()

if __name__ == "__main__":
    viewer = StreamViewer()
    viewer.start_viewing()