import math
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple

from stream_processor.infrastructure.neo4j_client import GraphStoreClient
from stream_processor.infrastructure.redis_client import TimeSeriesClient

logger = logging.getLogger(__name__)

class CresciHeuristicLabeler:
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", redis_host: str = "localhost"):
        self.graph_store = GraphStoreClient(uri=neo4j_uri)
        self.time_series = TimeSeriesClient(host=redis_host)

    def close(self):
        self.graph_store.close()

    def calculate_reputation(self, user_id: str) -> Tuple[float, int, int, int]:
        """
        Queries Neo4j directly to fetch the exact follower/following counts,
        returning the computed Social Reputation ratio and counts.
        Also returns the stored true_label for validation purposes.
        """
        query = """
        MATCH (u:User {id: $user_id})
        OPTIONAL MATCH (u)<-[:FOLLOWS]-(follower:User)
        WITH u, count(distinct follower) as followers
        OPTIONAL MATCH (u)-[:FOLLOWS]->(following:User)
        RETURN followers, count(distinct following) as following, u.true_label as true_label
        """
        try:
            with self.graph_store.driver.session() as session:
                result = session.run(query, user_id=user_id)
                record = result.single()
                if record:
                    followers = int(record["followers"])
                    following = int(record["following"])
                    true_label = record["true_label"]
                    
                    true_label = int(true_label) if true_label is not None else 0
                    
                    if followers + following == 0:
                        return 0.5, followers, following, true_label
                    
                    reputation = followers / (followers + following)
                    return reputation, followers, following, true_label
        except Exception as e:
            logger.error(f"Failed to query Neo4j for user reputation {user_id}: {e}")
        
        return 0.5, 0, 0, 0

    def calculate_temporal_entropy(self, user_id: str) -> Tuple[float, float, float]:
        """
        Fetches timeline action history from Redis, computes chronological deltas,
        and returns the raw Shannon Entropy, the normalized entropy, and the spam elements density.
        """
        key = f"user_timeline:{user_id}"
        try:
            records = self.time_series.redis.lrange(key, 0, -1)
            if not isinstance(records, list) or len(records) < 2:
                
                return 0.0, 0.0, 0.0
            
            parsed_records = []
            complex_count = 0
            
            for r in records:
                data = json.loads(r)
                parsed_records.append(data)
                if float(data.get("is_complex", 0.0)) > 0.0:
                    complex_count += 1
            
            spam_density = complex_count / len(parsed_records)
            
            
            timestamps = []
            for record in parsed_records:
                ts_str = record.get("ts")
                if ts_str:
                    try:
                        if ts_str.endswith("Z"):
                            ts_str = ts_str.replace("Z", "+00:00")
                        ts = datetime.fromisoformat(ts_str)
                        timestamps.append(ts)
                    except ValueError:
                        continue
            
            timestamps.sort()
            
            if len(timestamps) < 2:
                return 0.0, 0.0, spam_density
            
            
            deltas = []
            for i in range(1, len(timestamps)):
                delta = (timestamps[i] - timestamps[i-1]).total_seconds()
                deltas.append(max(delta, 0.0))
            
            # Bucket deltas into 5 scientific interval groups
            # B0: < 1s (sub-second bursts)
            # B1: 1s to 60s (high-velocity posts)
            # B2: 60s to 3600s (sub-hour scheduling)
            # B3: 3600s to 86400s (sub-day organic gap)
            # B4: >= 86400s (long intervals)
            buckets = [0] * 5
            for d in deltas:
                if d < 1.0:
                    buckets[0] += 1
                elif d < 60.0:
                    buckets[1] += 1
                elif d < 3600.0:
                    buckets[2] += 1
                elif d < 86400.0:
                    buckets[3] += 1
                else:
                    buckets[4] += 1
            
            total_deltas = len(deltas)
            entropy = 0.0
            for count in buckets:
                if count > 0:
                    p = count / total_deltas
                    entropy -= p * math.log2(p)
            
            
            max_possible_entropy = math.log2(5)
            normalized_entropy = entropy / max_possible_entropy
            
            return entropy, normalized_entropy, spam_density
            
        except Exception as e:
            logger.error(f"Failed to calculate entropy for user {user_id}: {e}")
            
        return 0.0, 0.0, 0.0

    def evaluate_user(self, user_id: str) -> Dict[str, Any]:
        """
        Calculates all three features and maps them to a weighted fuzzy heuristic observed label.
        """
        reputation, followers, following, true_label = self.calculate_reputation(user_id)
        entropy, normalized_entropy, spam_density = self.calculate_temporal_entropy(user_id)
        
        # Weighted fuzzy heuristic score:
        # High reputation (followers > friends) lowers score (more human)
        # High temporal entropy (irregular intervals) lowers score (more human)
        # High spam density (links, retweets) raises score (more bot)
        score = 0.35 * (1.0 - reputation) + 0.35 * (1.0 - normalized_entropy) + 0.30 * spam_density
        
        
        observed_label = 1 if score >= 0.60 else 0
        
        return {
            "user_id": user_id,
            "followers": followers,
            "following": following,
            "reputation": reputation,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "spam_density": spam_density,
            "heuristic_score": score,
            "observed_label": observed_label,
            "true_label": true_label
        }

    def build_dataset_v1(self, output_path: str = "data/processed/dataset_v1.csv") -> List[Dict[str, Any]]:
        """
        Scans all active users persisted in Neo4j, compiles their heuristic features,
        and saves them as a structured CSV/JSON file to build the V1 training dataset.
        """
        query = "MATCH (u:User) RETURN u.id as id"
        user_ids = []
        try:
            with self.graph_store.driver.session() as session:
                result = session.run(query)
                for record in result:
                    user_ids.append(record["id"])
        except Exception as e:
            logger.error(f"Failed to scan users in Neo4j: {e}")
            return []

        logger.info(f"Analyzing {len(user_ids)} active users for Dataset V1 compiling...")
        
        dataset = []
        for uid in user_ids:
            metrics = self.evaluate_user(uid)
            dataset.append(metrics)

        
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import pandas as pd
            df = pd.DataFrame(dataset)
            if output_path.endswith(".parquet"):
                df.to_parquet(output_path, index=False)
                logger.info(f"✅ Successfully compiled {len(dataset)} users in Parquet at: {output_path}")
            else:
                df.to_csv(output_path, index=False)
                logger.info(f"✅ Successfully compiled {len(dataset)} users in CSV at: {output_path}")
        except ImportError:
            
            import csv
            with open(out_file, "w", newline="", encoding="utf-8") as f:
                if dataset:
                    writer = csv.DictWriter(f, fieldnames=dataset[0].keys())
                    writer.writeheader()
                    writer.writerows(dataset)
            logger.info(f"✅ Successfully compiled {len(dataset)} users in fallback CSV format at: {output_path}")
            
        return dataset

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    labeler = CresciHeuristicLabeler()
    labeler.build_dataset_v1()
    labeler.close()
