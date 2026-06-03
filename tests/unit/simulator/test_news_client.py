import pytest
import urllib.request
from unittest.mock import MagicMock, patch
from simulator.infrastructure.news_client import GoogleNewsClient

MOCK_XML_SUCCESS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Breaking News: AI Agent Works Flawlessly</title>
    </item>
    <item>
      <title>Weather Update: Sunny Days Ahead</title>
    </item>
  </channel>
</rss>
"""

MOCK_XML_MISSING_TITLE_OR_TEXT = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <!-- Missing title completely -->
      <link>https://example.com</link>
    </item>
    <item>
      <title></title> <!-- Empty title -->
    </item>
    <item>
      <title>Valid Headline</title>
    </item>
  </channel>
</rss>
"""

def test_fetch_latest_headlines_success():
    # Mock urllib.request.urlopen context manager
    mock_response = MagicMock()
    mock_response.read.return_value = MOCK_XML_SUCCESS
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None
    
    with patch("urllib.request.urlopen", return_value=mock_response):
        client = GoogleNewsClient()
        result = client.fetch_latest_headlines(limit=2)
    
    assert "Breaking News: AI Agent Works Flawlessly" in result
    assert "Weather Update: Sunny Days Ahead" in result
    assert " | " in result

def test_fetch_latest_headlines_handles_missing_tags_safely():
    mock_response = MagicMock()
    mock_response.read.return_value = MOCK_XML_MISSING_TITLE_OR_TEXT
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None
    
    with patch("urllib.request.urlopen", return_value=mock_response):
        client = GoogleNewsClient()
        result = client.fetch_latest_headlines()
    
    # Empty title and missing title must be ignored safely, returning only the Valid Headline
    assert result == "Valid Headline"

def test_fetch_latest_headlines_graceful_fallback_on_error():
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        client = GoogleNewsClient()
        result = client.fetch_latest_headlines()
    
    assert result == "Standard day on the internet, no major breaking news."
