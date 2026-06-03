import pytest
import math
import uuid
from stream_processor.infrastructure.neo4j_client import GraphStoreClient
from stream_processor.infrastructure.redis_client import TimeSeriesClient
from stream_processor.main import StreamProcessorOrchestrator

def test_gnn_feature_extraction_pipeline():
    # 1. Initialize Clients
    graph_store = GraphStoreClient()
    time_series = TimeSeriesClient()
    orchestrator = StreamProcessorOrchestrator()
    
    # 2. Generate unique test IDs to isolate execution
    user_a = f"test-user-a-{uuid.uuid4().hex[:6]}"
    user_b = f"test-user-b-{uuid.uuid4().hex[:6]}"
    user_c = f"test-user-c-{uuid.uuid4().hex[:6]}"
    
    # 3. Simulate actions to build spatial graph topology
    # A follows B
    graph_store.update_topology({
        "user_id": user_a,
        "target_id": user_b,
        "action_type": "FOLLOW",
        "timestamp": "2026-06-02T00:00:01Z"
    })
    # B follows C
    graph_store.update_topology({
        "user_id": user_b,
        "target_id": user_c,
        "action_type": "FOLLOW",
        "timestamp": "2026-06-02T00:00:02Z"
    })
    # C follows A
    graph_store.update_topology({
        "user_id": user_c,
        "target_id": user_a,
        "action_type": "FOLLOW",
        "timestamp": "2026-06-02T00:00:03Z"
    })
    
    # 4. Simulate action logs for User A to build temporal features
    # Action 1: Standard short post (no tags/urls)
    time_series.record_action({
        "user_id": user_a,
        "content": "Short text.",
        "action_type": "POST",
        "timestamp": "2026-06-02T00:10:00Z"
    })
    # Action 2: Retweet containing a hashtag
    time_series.record_action({
        "user_id": user_a,
        "content": "RT @news Check this #breaking news!",
        "action_type": "REPLY",
        "timestamp": "2026-06-02T00:11:00Z"
    })
    
    # 5. Execute feature extraction
    features = orchestrator._extract_features(user_a)
    
    # Clean clients
    graph_store.close()
    
    # 6. Run validations and assertions
    print("\n--- GNN Feature Extraction Test Output ---")
    print("User A ID:", user_a)
    print("Target Node Index in sub-graph:", features["target_node_idx"])
    print("Temporal features sequence (shape [10, 2]):", features["temporal_features"])
    print("Node features in sub-graph:", features["node_features"])
    print("Edge index matrix:", features["edge_index"])
    
    # Temporal validations
    temporal = features["temporal_features"]
    assert len(temporal) == 10, "Temporal features must contain exactly 10 sequence items."
    
    # The latest action in Redis is Action 2 (since pipeline lpushes and newest is first)
    assert temporal[0][0] == min(len("RT @news Check this #breaking news!") / 280.0, 1.0)
    assert temporal[0][1] == 1.0, "Action 2 contains RT and hashtag; complexity flag must be 1.0"
    
    # The previous action is Action 1
    assert temporal[1][0] == min(len("Short text.") / 280.0, 1.0)
    assert temporal[1][1] == 0.0, "Action 1 is short and standard; complexity flag must be 0.0"
    
    # Remaining 8 items must be padded [0.0, 0.0]
    for idx in range(2, 10):
        assert temporal[idx] == [0.0, 0.0], f"Item {idx} is not padded correctly."
        
    # Spatial validations
    node_features = features["node_features"]
    edge_index = features["edge_index"]
    target_idx = features["target_node_idx"]
    
    assert len(node_features) >= 3, "Subgraph must contain at least the 3 connected test users."
    
    # Check that spatial stats for User A are correct:
    # A is followed by C (1 follower), and follows B (1 following)
    # So followers = 1, following = 1
    expected_log_followers = math.log1p(1.0)
    expected_log_friends = math.log1p(1.0)
    
    target_features = node_features[target_idx]
    assert abs(target_features[0] - expected_log_followers) < 1e-5, f"Expected log-followers {expected_log_followers}, got {target_features[0]}"
    assert abs(target_features[1] - expected_log_friends) < 1e-5, f"Expected log-friends {expected_log_friends}, got {target_features[1]}"
    
    # Verify edge index contains elements
    assert len(edge_index) == 2
    assert len(edge_index[0]) == len(edge_index[1])
    
    print("Database GNN Feature Engineering test passed successfully!")
