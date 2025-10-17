from datasets import load_dataset
import pandas as pd

def explore_datasets():
    """Test different Hugging Face news datasets"""
    datasets_to_test = [
        "cc_news",           # 700K+ general news
        "cnn_dailymail",     # 300K news articles
        "xsum",              # BBC summaries
        "newsroom",          # 1.3M articles
        "ag_news"            # 120K news articles
    ]
    
    for dataset_name in datasets_to_test:
        try:
            print(f"\n=== Testing {dataset_name} ===")
            dataset = load_dataset(dataset_name, trust_remote_code=True)
            
            # Show sample structure
            sample = dataset['train'][0]
            print(f"Columns: {list(sample.keys())}")
            print(f"Sample text: {str(sample)[:200]}...")
            
        except Exception as e:
            print(f"Error with {dataset_name}: {e}")

if __name__ == "__main__":
    explore_datasets()