import logging
import time
from typing import LiteralString
from confluent_kafka.admin import AdminClient, NewTopic
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class KafkaManager:
    def __init__(self, broker_url: str = "localhost:9092"):
        self.admin_client = AdminClient({"bootstrap.servers": broker_url})

    def create_topic(self, topic_name: str, num_partitions: int = 3, replication_factor: int = 1):
        topic_list = [NewTopic(topic_name, num_partitions, replication_factor)]
        
        futures = self.admin_client.create_topics(topic_list)
        for topic, future in futures.items():
            try:
                future.result()
                logger.info(f"Kafka topic '{topic}' created successfully.")
            except Exception as e:
                if "TopicExistsException" in str(e):
                    logger.info(f"Kafka topic '{topic}' already exists.")
                else:
                    logger.error(f"Failed to create topic '{topic}': {e}")

class Neo4jManager:
    def __init__(self, uri: str = "bolt://localhost:7687", auth: tuple = ("neo4j", "botdetection123")):
        self.driver = self._connect_with_retry(uri, auth)

    def _connect_with_retry(self, uri: str, auth: tuple, retries: int = 5, delay: int = 5):
        for attempt in range(retries):
            try:
                driver = GraphDatabase.driver(uri, auth=auth)
                driver.verify_connectivity()
                logger.info("Successfully connected to Neo4j.")
                return driver
            except ServiceUnavailable:
                logger.warning(f"Neo4j unavailable. Retrying in {delay}s (Attempt {attempt + 1}/{retries})...")
                time.sleep(delay)
        raise ConnectionError("Failed to connect to Neo4j after multiple retries.")

    def setup_constraints(self):
        queries: list[LiteralString] = [
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE INDEX user_created_at IF NOT EXISTS FOR (u:User) ON (u.created_at)"
        ]
        
        with self.driver.session() as session:
            for query in queries:
                session.run(query)
                logger.info(f"Executed Neo4j constraint/index: {query.split(' IF ')[0]}")

    def close(self):
        self.driver.close()

def main():
    logger.info("Initializing infrastructure setup...")
    
    kafka_manager = KafkaManager()
    kafka_manager.create_topic("user_actions")
    
    neo4j_manager = Neo4jManager()
    neo4j_manager.setup_constraints()
    neo4j_manager.close()
    
    logger.info("Infrastructure setup completed.")

if __name__ == "__main__":
    main()