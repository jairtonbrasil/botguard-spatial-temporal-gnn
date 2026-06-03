import urllib.request
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)

class GoogleNewsClient:
    def __init__(self, url: str = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"):
        self.url = url

    def fetch_latest_headlines(self, limit: int = 5) -> str:
        try:
            req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()
            
            root = ET.fromstring(xml_data)
            headlines = []
            for item in root.findall('./channel/item')[:limit]:
                title_elem = item.find('title')
                if title_elem is not None and title_elem.text is not None:
                    headlines.append(title_elem.text.strip())
            
            if not headlines:
                return "Standard day on the internet, no major breaking news."
                
            return " | ".join(headlines)
            
        except Exception as e:
            logger.error(f"Failed to fetch news context: {e}")
            return "Standard day on the internet, no major breaking news."