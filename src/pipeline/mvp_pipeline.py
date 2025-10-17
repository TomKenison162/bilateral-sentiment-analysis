import pandas as pd
from tqdm import tqdm
from src.data.hf_loader import HFDataLoader
from src.analysis.country_detector import CountryDetector
from src.analysis.sentiment_analyser import SentimentAnalyser

class MVPPipeline:
    def __init__(self):
        self.loader = HFDataLoader()
        self.country_detector = CountryDetector()
        self.sentiment_analyzer = SentimentAnalyser()
    
    def run_uk_usa_analysis(self, sample_size: int = 5000) -> pd.DataFrame:
        """Run MVP analysis: UK media coverage of USA"""
        print("Starting UK→USA Sentiment Analysis")
        
        # 1. Load data
        df = self.loader.load_sample(sample_size)
        
        # 2. Filter UK sources (simplified - using title/content analysis)
        print("Finding UK articles about USA...")
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = f"{row.get('title', '')} {row.get('text', '')}"
            
            # Check if article is about USA from UK perspective
            if self.country_detector.is_about_country_pair(text, 'uk', 'usa'):
                # Analyze sentiment
                sentiment = self.sentiment_analyzer.analyze_article(
                    row.get('title', ''),
                    row.get('text', '')[:1000]  # First 1000 chars for speed
                )
                
                results.append({
                    'title': row.get('title', ''),
                    'source_domain': row.get('domain', ''),
                    'text_preview': text[:200],
                    **sentiment
                })
        
        results_df = pd.DataFrame(results)
        print(f"✅ Found {len(results_df)} UK→USA articles")
        
        return results_df