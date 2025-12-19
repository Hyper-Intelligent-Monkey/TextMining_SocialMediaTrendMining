import os
import pandas as pd

def sample_tweets():
    csv_path = os.path.join(os.path.dirname(__file__), '../data/raw_tweets/dataset.csv')
    data = pd.read_csv(csv_path, low_memory=False)

    # Available dataset
    if 'content' in data.columns:
        df = data[['content']].rename(columns={'content': 'tweet'})
    elif 'tweet' in data.columns:
        df = data[['tweet']]
    else:
        raise KeyError("Neither 'content' nor 'tweet' column found in dataset.")

    # Limit to n rows (can be increased depending on the available dataset)
    df = df.iloc[:min(1000, len(df))].reset_index(drop=True)

    print(f" Loaded {len(df)} tweets")
    return df
