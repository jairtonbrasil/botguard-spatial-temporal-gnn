import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MachineLearningClient:
    """
    Client to communicate with the FastAPI Inference service.
    """
    def __init__(self, api_url: str = "http://127.0.0.1:8000"):
        self.api_url = api_url

    def evaluate_user(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sends the feature payload to the ML API and returns the decision.
        """
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=payload,
                timeout=2.0 
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to communicate with ML API: {e}")
            return None