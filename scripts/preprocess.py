import re
import nltk
from nltk.corpus import stopwords
import spacy
from textblob import Word
import os
import pandas as pd

# Stopwords + noise
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

noise_words = set([
    'amp', 'rt', 'lol', 'haha', 'hehe', 'omg', 'wow', 'ugh', 'hey',
    'ha', 'hehehe', 'hee', 'lmao'
])

all_noise = stop_words.union(noise_words)
repeating_pattern = re.compile(r'(lmao+|hee+|ha+|hihi+|lol+)', re.IGNORECASE)

# Load SpaCy
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Function to detect generic verbs
generic_verb_pos = {'AUX', 'VERB'}

def is_generic_verb(token):
    """
    Returns True if a token is a 'generic' or filler verb,
    e.g., auxiliaries or very common verbs like get, make, say.
    """
    if token.pos_ in generic_verb_pos:
        lemma = token.lemma_.lower()
        common_verbs = {'get', 'go', 'say', 'make', 'do', 'have', 'be'}
        if lemma in common_verbs:
            return True
        return False
    return False

def is_noise_word(token):
    token = token.lower()
    if token in all_noise:
        return True
    if repeating_pattern.fullmatch(token):
        return True
    if re.fullmatch(r'(.)\1{2,}', token):
        return True
    return False

INPUT_PATH = "data/results/text_summarized.csv"
OUTPUT_PATH = "data/cleaned/processed_summaries.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Cleaning
def clean_summarized_text(text):
    text = str(text).lower()
    
    # Extract hashtags and mentions
    hashtags = re.findall(r"#\w+", text)
    mentions = re.findall(r"@\w+", text)
    
    hashtags = [h.lower() for h in hashtags if not is_noise_word(h)]
    mentions = [m.lower() for m in mentions if not is_noise_word(m)]
    
    # Remove hashtags and mentions from main text
    text = re.sub(r"#\w+|@\w+", "", text)
    
    # Tokenize and lemmatize
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        if token.pos_ in ['NOUN', 'ADJ', 'PROPN', 'NUM', 'VERB'] \
                and not is_noise_word(lemma) \
                and not is_generic_verb(token):
            if token.pos_ == 'NUM':
                filtered_tokens.append(lemma)
            else:
                lemma = str(Word(lemma).correct())
                filtered_tokens.append(lemma)
    
    # Combine tokens with hashtags and mentions
    combined_tokens = filtered_tokens + hashtags + mentions
    combined_tokens = [t for t in combined_tokens if len(t) > 1]
    
    return ' '.join(combined_tokens)

df = pd.read_csv(INPUT_PATH)
df['processed_summary'] = df['summary'].apply(clean_summarized_text)
final_df = df[["tweet", "summary", "processed_summary"]]

# Save
final_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
print(f"Processed summaries saved")
