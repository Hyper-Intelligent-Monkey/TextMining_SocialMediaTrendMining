from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '../data/cleaned/processed_summaries.csv')
results_dir = os.path.join(BASE_DIR, '../data/results')
os.makedirs(results_dir, exist_ok=True)
results_csv_path = os.path.join(results_dir, 'trending_words.csv')

df = pd.read_csv(csv_path)
df['processed_summary'] = df['processed_summary'].fillna('').str.strip()
df = df[df['processed_summary'] != ''].reset_index(drop=True)

MIN_TOKEN_LEN = 3

# Token validation
def is_valid_token(tok: str) -> bool:
    if len(tok) < MIN_TOKEN_LEN:
        return False

    if tok.startswith('#') or tok.startswith('@'):
        return True

    if re.fullmatch(r"20(0[0-9]|1[0-9]|2[0-9])", tok):
        return True

    if re.fullmatch(r"[a-z0-9]+", tok):
        return True

    return False

# Custom tokenizer
def custom_tokenizer(text):
    raw_tokens = re.findall(
        r"(@[a-z0-9_]+|#[a-z0-9_]+|[a-z0-9%]+)",
        text.lower()
    )

    return [tok for tok in raw_tokens if is_valid_token(tok)]

# TF-IDF
vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, preprocessor=lambda x: x, token_pattern=None)

X = vectorizer.fit_transform(df['processed_summary'])

feature_names = vectorizer.get_feature_names_out()
tfidf_scores = X.sum(axis=0).A1

tfidf_df = pd.DataFrame({
    'word': feature_names,
    'score': tfidf_scores
})

# Frequency weighting
all_tokens = []
for text in df['processed_summary']:
    all_tokens.extend(custom_tokenizer(text))

word_freq = Counter(all_tokens)

tfidf_df['frequency'] = tfidf_df['word'].map(word_freq)
tfidf_df['trending_score'] = tfidf_df['score'] * tfidf_df['frequency']

tfidf_df = tfidf_df.sort_values(by='trending_score', ascending=False)

tfidf_df.to_csv(results_csv_path, index=False, encoding='utf-8')
print("Saved trending words ", results_csv_path)
