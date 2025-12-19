import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASE_DIR, "../data/results/clustered_trends.csv")
output_path = os.path.join(BASE_DIR, "../data/results/final_topics.csv")

# Load
df = pd.read_csv(input_path)

text_col = "processed_summary"
df = df[df[text_col].notna()].reset_index(drop=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Process clusters
final_clusters = []
for cluster_id, group in df.groupby("cluster_id"):
    texts = group[text_col].tolist()

    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    # Compute similarity
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    upper_tri = np.triu_indices_from(sim_matrix, k=1)
    avg_similarity = sim_matrix[upper_tri].mean()

    # Merge all keywords from this cluster
    merged_text = " ".join(texts)
    words = merged_text.split()

    top_keywords = [w for w, _ in Counter(words).most_common(15)]
    keywords_str = " ".join(top_keywords)

    if avg_similarity >= 0.4:
        final_clusters.append({
            "cluster_id": cluster_id,
            "semantic_cohesion": round(float(avg_similarity), 3),
            "tweet_count": len(texts),
            "ner_cleaned_tweets": keywords_str
        })


final_df = pd.DataFrame(final_clusters)
final_df = final_df.sort_values(by="semantic_cohesion", ascending=False)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)
