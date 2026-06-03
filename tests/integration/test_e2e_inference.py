import pytest
from ml_api.application.model_manager import InferenceManager

def test_gnn_inference_manager_direct():
    # 1. Initialize InferenceManager (bypassing starlette / version-sensitive TestClient)
    # The manager will warn and load random weights if the pt file does not exist, which is perfect for dry-runs!
    manager = InferenceManager()
    
    # 2. Assert device configuration
    assert manager.device is not None
    print(f"\nInitialized InferenceManager on device: {manager.device}")
    
    # 3. Simulate a GNN inference payload
    # temporal_features: shape [10, 2]
    temporal_features = [[0.1, 0.0] for _ in range(10)]
    
    # node_features: shape [3, 3] (3 nodes: log_followers, log_friends, log_ratio)
    node_features = [
        [1.2, 0.8, 0.4],
        [0.5, 1.5, 0.1],
        [2.0, 0.0, 2.0]
    ]
    
    # edge_index: shape [2, 6] (connections + self loops)
    edge_index = [
        [0, 1, 2, 0, 1, 2],
        [1, 2, 0, 0, 1, 2]
    ]
    
    # 4. Make a direct prediction call to PyTorch
    score = manager.predict(
        temporal_features=temporal_features,
        node_features=node_features,
        edge_index=edge_index,
        target_idx=0
    )
    
    # 5. Assertions on the score
    assert isinstance(score, float), "Score must be a floating point probability."
    assert 0.0 <= score <= 1.0, f"Score {score} is out of bounds [0, 1]."
    
    # 6. Apply calibration and tiered mitigations logic
    THRESHOLD_BAN = 0.90
    THRESHOLD_LIMIT = 0.70
    THRESHOLD_SAMPLE_LOWER = 0.40
    THRESHOLD_SAMPLE_UPPER = 0.60
    
    if score >= THRESHOLD_BAN:
        action = "BAN"
    elif score >= THRESHOLD_LIMIT:
        action = "LIMIT"
    else:
        action = "ALLOW"
        
    needs_review = THRESHOLD_SAMPLE_LOWER <= score <= THRESHOLD_SAMPLE_UPPER
    
    print("\n--- GNN Inference Manager E2E Prediction Output ---")
    print(f"Bot Probability:    {score:.6f}")
    print(f"Mitigation Action:  {action}")
    print(f"Needs Manual Review:{needs_review}")
    
    print("GNN Inference Manager integration test passed successfully!")
