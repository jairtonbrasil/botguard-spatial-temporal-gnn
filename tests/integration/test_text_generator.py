import pytest
import requests
from unittest.mock import MagicMock, patch
from simulator.infrastructure.text_generator import OllamaTextGenerator

def test_ollama_generator_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "This is a simulated tweet."}
    mock_response.raise_for_status.return_value = None
    
    with patch("requests.post", return_value=mock_response):
        generator = OllamaTextGenerator(model_name="dummy_model")
        result = generator.generate_text("Generate something")
    
    assert result == "This is a simulated tweet."

def test_ollama_generator_handles_connection_error():
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
        generator = OllamaTextGenerator(model_name="dummy_model")
        result = generator.generate_text("Generate something")
    
    assert result is None