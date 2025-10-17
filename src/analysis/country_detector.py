import re
from typing import List

class CountryDetector:
    def __init__(self):
        self.country_keywords = {
            'usa': ['usa', 'united states', 'america', 'american', 'washington', 'white house'],
            'uk': ['uk', 'united kingdom', 'britain', 'british', 'london', 'downing street'],
            'china': ['china', 'chinese', 'beijing'],
            'russia': ['russia', 'russian', 'moscow']
        }
    
    def detect_country_mentions(self, text: str) -> List[str]:
        """Detect which countries are mentioned in text"""
        mentioned_countries = []
        text_lower = text.lower()
        
        for country, keywords in self.country_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                mentioned_countries.append(country)
                
        return mentioned_countries
    
    def is_about_country_pair(self, text: str, source_country: str, target_country: str) -> bool:
        """Check if article is about specific country pair"""
        mentioned = self.detect_country_mentions(text)
        return source_country in mentioned and target_country in mentioned