import os
import pandas as pd
from transformers import pipeline, logging

logging.set_verbosity_error()

input_path = os.path.join(os.path.dirname(__file__), "../data/cleaned/simple_processed.csv")
df = pd.read_csv(input_path)

# Summarization model
summarizer = pipeline( task="summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")

summaries = []
for text in df["preprocessed_text"]:
    if not isinstance(text, str) or len(text.split()) < 15:
        summaries.append(text)
        continue

    result = summarizer(
        text,
        max_length=60,
        min_length=15,
        do_sample=False
    )
    summaries.append(result[0]["summary_text"])

df["summary"] = summaries

# Save
output_dir = os.path.join(os.path.dirname(__file__), "../data/results")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "text_summarized.csv")
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"Summarization completed")
