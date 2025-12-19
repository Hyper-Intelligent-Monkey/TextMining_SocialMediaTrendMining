import re
import os
import pandas as pd
from fetch_tweets import sample_tweets 

df = sample_tweets()

def simple_preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply preprocessing for each
df["preprocessed_text"] = df["tweet"].apply(simple_preprocess)

output_dir = os.path.join(os.path.dirname(__file__), "../data/cleaned")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "simple_processed.csv")

df[["tweet", "preprocessed_text"]].to_csv(
    output_path,
    index=False,
    encoding="utf-8"
)

print(f"Simple preprocessing completed")
