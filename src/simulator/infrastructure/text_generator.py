import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

class OllamaTextGenerator:
    def __init__(self, model_name: str = "phi3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate_text(self, prompt: str) -> Optional[str]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as error:
            logger.error(f"Failed to connect to Ollama API: {error}")
            return None