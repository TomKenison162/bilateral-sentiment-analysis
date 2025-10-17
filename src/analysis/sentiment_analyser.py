from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

class SentimentAnalyser:
    def __init__(self):
        # Use VADER for MVP (fast, good with news)
        self.vader = SentimentIntensityAnalyzer()
        
        # Could add transformer model later
        # self.transformer_sentiment = pipeline("sentiment-analysis")
    
    def analyze_article(self, title: str, text: str) -> dict:
        """Analyze sentiment of article"""
        combined_text = f"{title}. {text}"[:1000]  # Limit length
        
        # VADER analysis
        vader_scores = self.vader.polarity_scores(combined_text)
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'sentiment_label': self._get_sentiment_label(vader_scores['compound'])
        }
    
    def _get_sentiment_label(self, score: float) -> str:
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"
        

"""
Infomation gathered: 

Two working data sets
"""
