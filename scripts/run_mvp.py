import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.mvp_pipeline import MVPPipeline
import pandas as pd

def main():
    pipeline = MVPPipeline()
    
    # Run analysis
    results = pipeline.run_uk_usa_analysis(sample_size=10000)
    
    if len(results) > 0:
        # Basic analysis
        avg_sentiment = results['vader_compound'].mean()
        positive_pct = (results['sentiment_label'] == 'positive').mean() * 100
        negative_pct = (results['sentiment_label'] == 'negative').mean() * 100
        
        print(f"\n UK→USA Sentiment Results:")
        print(f"   Average Sentiment: {avg_sentiment:.3f}")
        print(f"   Positive Articles: {positive_pct:.1f}%")
        print(f"   Negative Articles: {negative_pct:.1f}%")
        print(f"   Neutral Articles: {100 - positive_pct - negative_pct:.1f}%")
        
        # Save results
        results.to_csv('data/mvp_results.csv', index=False)
        print(f"Saved {len(results)} articles to data/mvp_results.csv")
        
        # Show sample
        print(f"\nSample Articles:")
        for idx, row in results.head(3).iterrows():
            print(f"   - {row['title'][:60]}... → {row['sentiment_label']} ({row['vader_compound']:.3f})")
    else:
        print("❌ No UK→USA articles found in sample")

if __name__ == "__main__":
    main()