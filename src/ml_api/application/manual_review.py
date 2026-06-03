import json
import csv
import sys
from pathlib import Path
import redis

class ActiveLearningReviewer:
    def __init__(self, redis_host: str = "localhost", output_path: str = "data/processed/manual_labels.csv"):
        self.redis = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        self.output_file = Path(output_path)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def load_queue(self, limit: int = 50) -> list:
        try:
            raw_items = self.redis.lrange("active_learning:queue", 0, limit - 1)
            if not isinstance(raw_items, list):
                return []
            return [json.loads(item) for item in raw_items]
        except redis.RedisError as e:
            print(f"\033[91mFailed to connect to Redis: {e}\033[0m")
            return []

    def save_label(self, event: dict, label: int):
        file_exists = self.output_file.exists()
        with open(self.output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["user_id", "bot_probability", "labeled_as", "content", "features"])
            
            writer.writerow([
                event["user_id"],
                event["bot_probability"],
                label,
                event["content"],
                json.dumps(event["features"])
            ])

    def remove_from_queue(self, event_str: str):
        try:
            self.redis.lrem("active_learning:queue", 1, event_str)
        except redis.RedisError:
            pass

    def run(self):
        print("\033[94m====================================================\033[0m")
        print("\033[94m       BOTGUARD ACTIVE LEARNING AUDIT CLI           \033[0m")
        print("\033[94m====================================================\033[0m")
        
        raw_items = self.redis.lrange("active_learning:queue", 0, -1)
        if not isinstance(raw_items, list) or not raw_items:
            print("\033[93mNo uncertain events found in Redis queue.\033[0m")
            return

        events = [json.loads(item) for item in raw_items]
        print(f"\033[92mFound {len(events)} events awaiting human annotation.\033[0m\n")

        labeled_count = 0
        for raw_str, event in zip(raw_items, events):
            print("\033[90m----------------------------------------------------\033[0m")
            print(f"👤 User ID: \033[1m{event['user_id']}\033[0m")
            print(f"📈 Model Bot Probability: \033[93m{event['bot_probability']:.4f}\033[0m")
            print(f"🕒 Timestamp: {event['timestamp']}")
            print(f"💬 Text Content: \033[3m\"{event['content']}\"\033[0m\n")

            while True:
                choice = input("Assign Label -> [b]ot / [h]uman / [s]kip / [q]uit: ").strip().lower()
                if choice in ["b", "bot", "1"]:
                    self.save_label(event, 1)
                    self.remove_from_queue(raw_str)
                    print("\033[91mLabeled as BOT.\033[0m")
                    labeled_count += 1
                    break
                elif choice in ["h", "human", "0"]:
                    self.save_label(event, 0)
                    self.remove_from_queue(raw_str)
                    print("\033[92mLabeled as HUMAN.\033[0m")
                    labeled_count += 1
                    break
                elif choice in ["s", "skip"]:
                    print("Event skipped.")
                    break
                elif choice in ["q", "quit"]:
                    print(f"\nAudit session ended. Total new annotations saved: {labeled_count}")
                    sys.exit(0)
                else:
                    print("Invalid input. Use 'b' for Bot, 'h' for Human, 's' to Skip, or 'q' to Quit.")

        print(f"\nAudit completed! Total new annotations saved: {labeled_count}")

if __name__ == "__main__":
    reviewer = ActiveLearningReviewer()
    reviewer.run()
