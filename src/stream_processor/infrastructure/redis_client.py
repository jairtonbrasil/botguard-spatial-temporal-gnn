import json
import logging
import redis

logger = logging.getLogger(__name__)

class TimeSeriesClient:
    def __init__(self, host: str = "localhost", port: int = 6379, max_history: int = 100):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)
        self.max_history = max_history

    def record_action(self, action_data: dict):
        user_id = action_data.get("user_id")
        if not user_id:
            return

        key = f"user_timeline:{user_id}"
        
        content = action_data.get("content") or ""
        action_type = action_data.get("action_type") or "POST"
        

        len_feat = min(len(content) / 280.0, 1.0)
        is_complex = 1.0 if (content.startswith("RT @") or "http" in content or "#" in content) else 0.0
        
        event_record = {
            "ts": action_data.get("timestamp"),
            "type": action_type,
            "len_feat": len_feat,
            "is_complex": is_complex
        }
        
        try:
            pipeline = self.redis.pipeline()
            pipeline.lpush(key, json.dumps(event_record))
            pipeline.ltrim(key, 0, self.max_history - 1)
            pipeline.expire(key, 86400) 
            pipeline.execute()
        except redis.RedisError as e:
            logger.error(f"Redis pipeline execution failed: {e}")

    def get_timeline_features(self, user_id: str, limit: int = 10) -> list:
        """
        Retrieves the latest 'limit' temporal features for a user.
        Pads sequence with [0.0, 0.0] if there are fewer than 'limit' actions.
        """
        key = f"user_timeline:{user_id}"
        try:
            records = self.redis.lrange(key, 0, limit - 1)
            if not isinstance(records, list):
                return [[0.0, 0.0]] * limit
                
            features = []
            for r in records:
                data = json.loads(r)
                features.append([
                    float(data.get("len_feat", 0.0)),
                    float(data.get("is_complex", 0.0))
                ])
            while len(features) < limit:
                features.append([0.0, 0.0])
            return features
        except redis.RedisError as e:
            logger.error(f"Failed to fetch timeline from Redis for user {user_id}: {e}")
            return [[0.0, 0.0]] * limit