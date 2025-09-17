# -------------------------------------------------------------
# Feature Extraction: Bag-of-Words (BoW) & TF-IDF
# Libraries: Scikit-learn, Pandas
# -------------------------------------------------------------

# Install required libraries if needed:
# pip install scikit-learn pandas nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download stopwords (first time only)
nltk.download("stopwords")

# -------------------------
# Example dataset (Movie reviews)
# -------------------------
corpus = [
    "I loved the movie, it was fantastic and exciting!",
    "The film was boring and too long.",
    "Absolutely wonderful acting, but the story was predictable.",
    "I did not enjoy the movie, it was disappointing.",
    "Great direction and good storyline, I liked it."
]

print("===== Original Corpus =====")
for doc in corpus:
    print(doc)

# -------------------------
# 1. Bag-of-Words (BoW)
# -------------------------
print("\n===== Bag-of-Words Representation =====")

bow_vectorizer = CountVectorizer(stop_words=stopwords.words("english"))
bow_matrix = bow_vectorizer.fit_transform(corpus)

# Show Vocabulary
print("Vocabulary:\n", bow_vectorizer.get_feature_names_out())

# Convert to DataFrame for readability
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
print("\nBoW Table:\n", bow_df)

# -------------------------
# 2. TF-IDF Representation
# -------------------------
print("\n===== TF-IDF Representation =====")

tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Show Vocabulary
print("Vocabulary:\n", tfidf_vectorizer.get_feature_names_out())

# Convert to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Table:\n", tfidf_df.round(2))   # rounded for clarity
