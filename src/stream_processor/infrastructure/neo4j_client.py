import logging
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class GraphStoreClient:
    def __init__(self, uri: str = "bolt://localhost:7687", auth: tuple = ("neo4j", "botdetection123")):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def update_topology(self, action_data: dict):
        action_type = action_data.get("action_type")
        true_label = action_data.get("true_label")
        
        if action_type == "POST":
            self._upsert_user(action_data["user_id"], action_data["timestamp"], true_label)
        elif action_type in ["REPLY", "RETWEET", "FOLLOW"]:
            self._upsert_interaction(
                source_id=action_data["user_id"],
                target_id=action_data["target_id"],
                action_type=action_type,
                timestamp=action_data["timestamp"],
                true_label=true_label
            )
        else:
            logger.warning(f"Unknown action type for graph topology: {action_type}")

    def _upsert_user(self, user_id: str, timestamp: str, true_label: int | None = None):
        if true_label is not None:
            query = """
            MERGE (u:User {id: $user_id})
            ON CREATE SET u.created_at = $timestamp, u.last_active = $timestamp, u.true_label = $true_label
            ON MATCH SET u.last_active = $timestamp, u.true_label = $true_label
            """
        else:
            query = """
            MERGE (u:User {id: $user_id})
            ON CREATE SET u.created_at = $timestamp, u.last_active = $timestamp
            ON MATCH SET u.last_active = $timestamp
            """
        self._execute_write(query, user_id=user_id, timestamp=timestamp, true_label=true_label)

    def _upsert_interaction(self, source_id: str, target_id: str, action_type: str, timestamp: str, true_label: int | None = None):
        if not target_id:
            return

        relationship_map = {
            "REPLY": "REPLIES_TO",
            "RETWEET": "RETWEETS",
            "FOLLOW": "FOLLOWS"
        }
        rel_type = relationship_map.get(action_type)

        if true_label is not None:
            query = f"""
            MERGE (source:User {{id: $source_id}})
            ON CREATE SET source.created_at = $timestamp, source.last_active = $timestamp, source.true_label = $true_label
            ON MATCH SET source.last_active = $timestamp, source.true_label = $true_label
            
            MERGE (target:User {{id: $target_id}})
            
            MERGE (source)-[r:{rel_type}]->(target)
            ON CREATE SET r.count = 1, r.last_interaction = $timestamp
            ON MATCH SET r.count = r.count + 1, r.last_interaction = $timestamp
            """
        else:
            query = f"""
            MERGE (source:User {{id: $source_id}})
            ON CREATE SET source.created_at = $timestamp, source.last_active = $timestamp
            ON MATCH SET source.last_active = $timestamp
            
            MERGE (target:User {{id: $target_id}})
            
            MERGE (source)-[r:{rel_type}]->(target)
            ON CREATE SET r.count = 1, r.last_interaction = $timestamp
            ON MATCH SET r.count = r.count + 1, r.last_interaction = $timestamp
            """
        self._execute_write(query, source_id=source_id, target_id=target_id, timestamp=timestamp, true_label=true_label)

    def _execute_write(self, query: str, **parameters):
        try:
            with self.driver.session() as session:
                session.run(query, parameters)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to execute Cypher query: {e}")

    def get_subgraph_features(self, target_user_id: str) -> dict:
        """
        Retrieves the 1-hop neighborhood sub-graph around target_user_id,
        computes spatial node features (followers, following, log-transformed ratio),
        and formats the edge_index tensor indices for PyTorch GNN inference.
        """
        import math
        
        # Cypher query to retrieve the target user and their 1-hop neighborhood
        query = """
        MATCH (target:User {id: $user_id})
        OPTIONAL MATCH (target)-[r1]-(neighbor:User)
        WITH target, collect(distinct neighbor) + target AS nodes
        
        UNWIND nodes AS n
        OPTIONAL MATCH (n)<-[:FOLLOWS]-(follower:User)
        WITH nodes, n, count(distinct follower) AS followers_count
        OPTIONAL MATCH (n)-[:FOLLOWS]->(following:User)
        WITH nodes, n, followers_count, count(distinct following) AS following_count
        
        WITH nodes, collect({
            id: n.id,
            followers: followers_count,
            following: following_count
        }) AS node_stats
        
        UNWIND nodes AS source
        UNWIND nodes AS target_node
        MATCH (source)-[r]->(target_node)
        RETURN node_stats, collect(distinct {
            source: source.id,
            target: target_node.id,
            type: type(r)
        }) AS edges
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, user_id=target_user_id)
                record = result.single()
                
                if not record or not record["node_stats"]:
                    
                    return {
                        "target_node_idx": 0,
                        "node_features": [[0.0, 0.0, 0.0]],
                        "edge_index": [[0], [0]]
                    }
                
                node_stats = record["node_stats"]
                edges = record["edges"] or []
                
                # Map node IDs to unique indices
                id_to_index = {node["id"]: idx for idx, node in enumerate(node_stats)}
                
                target_node_idx = id_to_index.get(target_user_id, 0)
                
                # Compute spatial node features
                node_features = []
                for node in node_stats:
                    followers = float(node["followers"])
                    following = float(node["following"])
                    
                    log_followers = math.log1p(followers)
                    log_friends = math.log1p(following)
                    
                    ratio = followers / (following + 1.0)
                    log_ratio = math.log1p(ratio)
                    
                    node_features.append([log_followers, log_friends, log_ratio])
                
                # Map edges to indexed graph structure
                edge_starts = []
                edge_ends = []
                for edge in edges:
                    source_idx = id_to_index.get(edge["source"])
                    target_idx = id_to_index.get(edge["target"])
                    if source_idx is not None and target_idx is not None:
                        edge_starts.append(source_idx)
                        edge_ends.append(target_idx)
                
                # Add self-loops to guarantee graph connectivity and SAGEConv stability
                for idx in range(len(node_stats)):
                    edge_starts.append(idx)
                    edge_ends.append(idx)
                
                return {
                    "target_node_idx": target_node_idx,
                    "node_features": node_features,
                    "edge_index": [edge_starts, edge_ends]
                }
                
        except Exception as e:
            logger.error(f"Failed to retrieve subgraph features for user {target_user_id}: {e}")
            return {
                "target_node_idx": 0,
                "node_features": [[0.0, 0.0, 0.0]],
                "edge_index": [[0], [0]]
            }