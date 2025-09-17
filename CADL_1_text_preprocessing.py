# -------------------------------------------------------------
# Text Pre-processing using NLTK and spaCy
# Steps: Tokenization, Stemming, Lemmatization, Stopword Removal
# -------------------------------------------------------------

# Install required libraries (only run once if not installed)
# pip install nltk spacy

import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Download NLTK data (only needed first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')   # optional: improves lemmatizer

# Load spaCy English model
# run in terminal once if missing: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Example text corpus
# -------------------------
corpus = [
    "The economy is showing signs of recovery after the pandemic.",
    "Natural Language Processing (NLP) makes machines understand human language.",
    "Many people are posting their opinions on social media platforms every day."
]

print("===== Original Corpus =====")
for doc in corpus:
    print(doc)

# -------------------------
# 1. TOKENIZATION
# -------------------------
print("\n===== Tokenization =====")
for doc in corpus:
    # NLTK Tokenization
    nltk_tokens = word_tokenize(doc)
    
    # spaCy Tokenization
    spacy_doc = nlp(doc)
    spacy_tokens = [token.text for token in spacy_doc]

    print(f"\nOriginal: {doc}")
    print(f"[NLTK] Tokens: {nltk_tokens}")
    print(f"[spaCy] Tokens: {spacy_tokens}")

# -------------------------
# 2. STEMMING (NLTK ONLY)
# -------------------------
print("\n===== Stemming =====")
porter = PorterStemmer()
snowball = SnowballStemmer("english")

for doc in corpus:
    tokens = word_tokenize(doc)  # NLTK tokens
    porter_stems = [porter.stem(w) for w in tokens]
    snowball_stems = [snowball.stem(w) for w in tokens]

    print(f"\nOriginal: {doc}")
    print(f"[NLTK] Porter Stemmer: {porter_stems}")
    print(f"[NLTK] Snowball Stemmer: {snowball_stems}")

# -------------------------
# 3. LEMMATIZATION
# -------------------------
print("\n===== Lemmatization =====")
lemmatizer = WordNetLemmatizer()

for doc in corpus:
    tokens = word_tokenize(doc)

    # NLTK Lemmatization
    nltk_lemmas = [lemmatizer.lemmatize(w) for w in tokens]

    # spaCy Lemmatization
    spacy_doc = nlp(doc)
    spacy_lemmas = [token.lemma_ for token in spacy_doc]

    print(f"\nOriginal: {doc}")
    print(f"[NLTK] Lemmatizer: {nltk_lemmas}")
    print(f"[spaCy] Lemmatizer: {spacy_lemmas}")

# -------------------------
# 4. STOPWORD REMOVAL
# -------------------------
print("\n===== Stopword Removal =====")
stop_words = set(stopwords.words('english'))

for doc in corpus:
    # NLTK Stopword Removal
    tokens = word_tokenize(doc.lower())
    filtered_tokens_nltk = [w for w in tokens if w not in stop_words and w.isalpha()]

    # spaCy Stopword Removal
    spacy_doc = nlp(doc)
    filtered_tokens_spacy = [token.text for token in spacy_doc if not token.is_stop and token.is_alpha]

    print(f"\nOriginal: {doc}")
    print(f"[NLTK] After Stopword Removal: {filtered_tokens_nltk}")
    print(f"[spaCy] After Stopword Removal: {filtered_tokens_spacy}")
