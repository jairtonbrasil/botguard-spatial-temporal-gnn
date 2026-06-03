import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from ml_api.application.model_manager import InferenceManager
import logging

logger = logging.getLogger(__name__)

class InferenceRequest(BaseModel):
    user_id: str
    target_node_idx: int = Field(..., description="Index of the target user in the node_features array.")
    temporal_features: List[List[float]] = Field(..., description="Sequence of recent user actions.")
    node_features: List[List[float]] = Field(..., description="Feature matrix for all nodes in the sub-graph.")
    edge_index: List[List[int]] = Field(..., description="Edge connectivity matrix [2, num_edges].")

class ShadowLoadRequest(BaseModel):
    model_path: str = Field("data/weights/twibot_shadow.pt", description="Path to shadow weights file")

app = FastAPI(title="BotGuard Inference API", version="1.2.0")

model_manager = InferenceManager(model_path="data/weights/twibot_baseline.pt")

THRESHOLD_BAN = 0.90
THRESHOLD_LIMIT = 0.70
THRESHOLD_SAMPLE_LOWER = 0.30
THRESHOLD_SAMPLE_UPPER = 0.80

def get_action_for_score(score: float) -> str:
    if score >= THRESHOLD_BAN:
        return "BAN"
    elif score >= THRESHOLD_LIMIT:
        return "LIMIT"
    return "ALLOW"

@app.get("/health")
def health_check():
    return {"status": "online", "device": str(model_manager.device)}

@app.post("/predict")
def predict_bot(payload: InferenceRequest):
    if not payload.node_features or not payload.temporal_features:
        raise HTTPException(status_code=400, detail="Feature arrays cannot be empty.")
        
    try:
        score = model_manager.predict(
            temporal_features=payload.temporal_features,
            node_features=payload.node_features,
            edge_index=payload.edge_index,
            target_idx=payload.target_node_idx
        )
        action = get_action_for_score(score)
        needs_review = THRESHOLD_SAMPLE_LOWER <= score <= THRESHOLD_SAMPLE_UPPER
        
        explanation = model_manager.explain(
            temporal_features=payload.temporal_features,
            node_features=payload.node_features,
            edge_index=payload.edge_index,
            target_idx=payload.target_node_idx
        )
        
        shadow_inference = None
        if model_manager.shadow_model is not None:
            shadow_score = model_manager.predict_shadow(
                temporal_features=payload.temporal_features,
                node_features=payload.node_features,
                edge_index=payload.edge_index,
                target_idx=payload.target_node_idx
            )
            if shadow_score is not None:
                shadow_action = get_action_for_score(shadow_score)
                model_manager.update_shadow_stats(score, shadow_score, action, shadow_action)
                shadow_inference = {
                    "bot_probability": shadow_score,
                    "action": shadow_action,
                    "diverged": shadow_action != action
                }

        response_payload = {
            "user_id": payload.user_id,
            "bot_probability": score,
            "action": action,
            "needs_manual_review": needs_review,
            "explanation": explanation,
            "shadow_inference": shadow_inference
        }
        
        if needs_review:
            logger.info(f"SAMPLING: User {payload.user_id[:8]} queued for review. Score: {score:.4f}")
            
        return response_payload

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
def reload_model():
    try:
        model_manager.reload("data/weights/twibot_baseline.pt")
        return {"status": "success", "message": "Model weights reloaded successfully in-memory."}
    except Exception as e:
        logger.error(f"Failed to reload model weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/shadow/load")
def load_shadow_model(payload: ShadowLoadRequest):
    try:
        model_manager.load_shadow(payload.model_path)
        if not model_manager.shadow_model:
            raise HTTPException(status_code=400, detail="Failed to load shadow model weights.")
        return {"status": "success", "message": f"Shadow model loaded from {payload.model_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/shadow/stats")
def get_shadow_stats():
    return model_manager.get_shadow_report()

@app.post("/shadow/promote")
def promote_shadow_model():
    try:
        model_manager.promote_shadow()
        return {"status": "success", "message": "Shadow model successfully promoted to production."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))