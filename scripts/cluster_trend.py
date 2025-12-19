import pandas as pd
import itertools
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
processed_path = os.path.join(BASE_DIR, "../data/cleaned/processed_summaries.csv")
trending_path = os.path.join(BASE_DIR, "../data/results/trending_words.csv")
output_path = os.path.join(BASE_DIR, "../data/results/clustered_trends.csv")

tweets_df = pd.read_csv(processed_path)
trending_df = pd.read_csv(trending_path)
scores = dict(zip(trending_df["word"], trending_df["score"]))

# Custom tokenizer
def tokenize(text):
    return re.findall(r"[A-Za-z0-9@#%]+", str(text).lower())

# Compute total trending score for a tweet
def get_tweet_score(words):
    return sum(scores.get(w, 0) for w in words)

# Preprocess tweets and compute scores
tweet_tokens = []
for _, row in tweets_df.iterrows():
    text = str(row.get("processed_summary", ""))
    original_text = str(row.get("summary", ""))
    words = set(tokenize(text))
    score = get_tweet_score(words)
    if score > 1:
        tweet_tokens.append({
            "original_summary": original_text,
            "processed_summary": text,
            "words": words,
            "score": score
        })

tweet_tokens.sort(key=lambda x: x["score"], reverse=True)

# Cluster tweets based on shared trending words
clusters = []
used = set()

for i, a in enumerate(tweet_tokens):
    if i in used: continue
    cluster = [a]
    for j, b in enumerate(tweet_tokens[i + 1:], start=i + 1):
        if j in used: continue
        if len(a["words"] & b["words"]) >= 2:  # minimum shared words
            cluster.append(b)
            used.add(j)

    # Merge strongly connected tweets
    merged = []
    for t1, t2 in itertools.combinations(cluster, 2):
        if len(t1["words"] & t2["words"]) >= 3:
            merged.append(t2)
    cluster.extend(merged)

    clusters.append(cluster)
    ''''''
    used.add(i)

# Keep only clusters with at least 3 tweets
filtered_clusters = [c for c in clusters if len(c) >= 3]

# Prepare output
output_clusters = []
for idx, cluster in enumerate(filtered_clusters, start=1):
    for item in cluster:
        output_clusters.append({
            "cluster_id": idx,
            "original_summary": item["original_summary"],
            "processed_summary": item["processed_summary"],
            "score": item["score"]
        })

# Save clusters
cluster_df = pd.DataFrame(output_clusters)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cluster_df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Clustered trends saved")
