import pytest
import json
import random
import time
from confluent_kafka import Consumer, KafkaError
from simulator.main import SimulatorOrchestrator

def test_simulator_cycle_publishes_to_kafka():
    # 1. Create a consumer first with latest offset reset
    unique_group = f"test-simulator-group-{random.randint(10000, 99999)}"
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': unique_group,
        'auto.offset.reset': 'latest',
        'enable.auto.commit': False
    })
    consumer.subscribe(['user_actions'])
    
    # Force consumer to connect and register with the broker by polling once (returns None)
    consumer.poll(0.5)
    
    # 2. Instantiate orchestrator and run a single cycle
    orchestrator = SimulatorOrchestrator()
    orchestrator.run_cycle()
    
    # 3. Poll for the newly published message
    msg = None
    retries = 15
    while retries > 0:
        msg = consumer.poll(1.0)
        if msg is not None:
            if not msg.error():
                break
            elif msg.error().code() == KafkaError._PARTITION_EOF:
                msg = None
        retries -= 1
        
    consumer.close()
    
    # 4. Assertions on the published event
    assert msg is not None, "Failed to retrieve the published action from Kafka topic 'user_actions'."
    
    payload = json.loads(msg.value().decode('utf-8'))
    print("\n[TEST SUCCESS] Consumed newly generated event payload:", payload)
    
    assert "user_id" in payload, "Missing 'user_id' in action payload"
    assert "action_type" in payload, "Missing 'action_type' in action payload"
    assert "timestamp" in payload, "Missing 'timestamp' in action payload"
    assert "true_label" in payload, "Missing 'true_label' in action payload"
    assert payload["true_label"] in [0, 1], "Invalid 'true_label' value"
    
    if payload["action_type"] in ["POST", "REPLY"]:
        assert "content" in payload and payload["content"], f"Missing text content for action {payload['action_type']}"
        # We assert that the LLM generated real text instead of returning the fallback error string
        assert payload["content"] != "Failed to generate text.", "Ollama failed to generate text, returning the fallback error."
    
    print("Integration test passed successfully!")
