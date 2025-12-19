import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASE_DIR, "../data/results/final_topics.csv")
df = pd.read_csv(input_path)

# Convert to numeric columns
df['cluster_id'] = pd.to_numeric(df['cluster_id'], errors='coerce')
df['semantic_cohesion'] = pd.to_numeric(df['semantic_cohesion'], errors='coerce')
df['tweet_count'] = pd.to_numeric(df['tweet_count'], errors='coerce')

# Bar Chart
df_sorted = df.sort_values(by='semantic_cohesion', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_sorted,
    x='cluster_id',
    y='semantic_cohesion',
    palette='coolwarm'
)
plt.title("Clusters Ranked by Semantic Cohesion")
plt.xlabel("Cluster ID")
plt.ylabel("Semantic Cohesion")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "../data/visual_results/bar_chart.png"))
plt.close()

# Bubble Chart
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='tweet_count',
    y='semantic_cohesion',
    size='tweet_count',
    hue='semantic_cohesion',
    sizes=(50, 1000),
    alpha=0.6,
    palette='plasma',
    legend=False
)
for i, row in df.iterrows():
    plt.text(row['tweet_count'], row['semantic_cohesion'], str(row['cluster_id']), fontsize=13)
plt.title("Bubble Chart — Popularity vs Semantic Cohesion")
plt.xlabel("Tweet Count")
plt.ylabel("Semantic Cohesion")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "../data/visual_results/bubble_chart.png"))
plt.close()

# Word Clouds
for _, row in df_sorted.iterrows():
    text = str(row['ner_cleaned_tweets'])
    
    # Including hashtags and mentions
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b|#[a-zA-Z0-9]+|@[a-zA-Z0-9]+', text)
    filtered_text = ' '.join(tokens)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate(filtered_text)

    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Cluster {row['cluster_id']}")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"../data/visual_results/wordCloud/wordcloud_cluster_{row['cluster_id']}.png"))
    plt.close()
