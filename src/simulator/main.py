import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import time
import random
import json
import logging
from simulator.domain.agents import NormalUserAgent, BotAgent
from simulator.domain.prompts import build_human_prompt, build_bot_prompt
from simulator.infrastructure.news_client import GoogleNewsClient
from simulator.infrastructure.text_generator import OllamaTextGenerator
from simulator.infrastructure.kafka_producer import EventProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SimulatorOrchestrator:
    def __init__(self):
        self.news_client = GoogleNewsClient()
        self.llm_generator = OllamaTextGenerator(model_name="phi3") 
        self.kafka_producer = EventProducer()
        
        self.active_users = []
        self.active_bots = []

    def _get_random_target(self):
        """Allows agents to interact with users already present in the simulation."""
        if not self.active_users:
            return None
        return random.choice(self.active_users).user_id

    def run_cycle(self):
        """Executes a single simulation cycle: agent selection, text generation, and publishing."""
        news_context = self.news_client.fetch_latest_headlines()
        
        # 80% chance for a human action, 20% chance for a bot action
        is_bot = random.random() < 0.20
        
        if is_bot:
            if self.active_bots and random.random() < 0.7:
                agent = random.choice(self.active_bots)
            else:
                agent = BotAgent()
                self.active_bots.append(agent)
            
            prompt = build_bot_prompt(news_context)
            target = self._get_random_target()
        else:
            if self.active_users and random.random() < 0.7:
                agent = random.choice(self.active_users)
            else:
                agent = NormalUserAgent()
                self.active_users.append(agent)
                
            prompt = build_human_prompt(news_context)
            target = self._get_random_target()

        # Generate text using RAG context + Local LLM
        content = self.llm_generator.generate_text(prompt)
        if not content:
            content = "Failed to generate text."

        
        action = agent.generate_action(content=content, target_id=target)
        
        
        action_payload = json.loads(action.model_dump_json(exclude={'true_label'}))
        action_payload['true_label'] = action.true_label 

        self.kafka_producer.publish_action(action_payload)
        logger.info(f"Generated [Bot: {is_bot}] - Action: {action.action_type} - Target: {target}")

    def run_continuously(self, delay_seconds: float = 3.0):
        logger.info("Starting simulation loop... Press Ctrl+C to stop.")
        try:
            while True:
                self.run_cycle()
                time.sleep(delay_seconds)
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user.")
        finally:
            self.kafka_producer.flush()

if __name__ == "__main__":
    orchestrator = SimulatorOrchestrator()
    orchestrator.run_continuously(delay_seconds=3.0)