from datasets import load_dataset
import pandas as pd
import re
from typing import List, Dict

class HFDataLoader:
    def __init__(self, dataset_name: str = "cc_news"):
        self.dataset_name = dataset_name
        self.dataset = None
        
    def load_sample(self, sample_size: int = 1000) -> pd.DataFrame:
        """Load a sample of the dataset for development"""
        print("Loading dataset...")
        self.dataset = load_dataset(self.dataset_name, split=f'train[:{sample_size}]')
        df = pd.DataFrame(self.dataset)
        print(f"Loaded {len(df)} articles")
        return df
    
    def filter_by_domain(self, df: pd.DataFrame, domains: List[str]) -> pd.DataFrame:
        """Filter articles by source domain"""
        # cc_news has 'domain' column - use regex to match domains
        pattern = '|'.join(domains)
        mask = df['domain'].str.contains(pattern, case=False, na=False)
        return df[mask]

# Domain mappings for future expansion
UK_DOMAINS = ['bbc.co.uk', 'theguardian.com', 'dailymail.co.uk', 'telegraph.co.uk']
USA_DOMAINS = ['cnn.com', 'foxnews.com', 'nytimes.com', 'washingtonpost.com']
