# -------------------------------------------------------------
# Topic Modeling with LDA (Latent Dirichlet Allocation)
# Libraries: NLTK, Gensim, pyLDAvis
# -------------------------------------------------------------

# Install if missing:
# pip install nltk gensim pyLDAvis

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# -------------------------
# Example text corpus
# -------------------------
corpus = [
    "The economy is recovering after the pandemic with growth in the IT sector.",
    "Artificial Intelligence and Machine Learning are transforming industries.",
    "Climate change is a pressing global issue that needs immediate attention.",
    "Social media platforms are influencing the opinions of young people.",
    "Healthcare systems are adopting digital technologies for better patient care.",
    "New advancements in space exploration are capturing the world's attention."
]

print("===== Original Corpus =====")
for doc in corpus:
    print(doc)

# -------------------------
# 1. Preprocessing
# -------------------------
stop_words = set(stopwords.words("english"))
processed_corpus = []

for doc in corpus:
    tokens = word_tokenize(doc.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    processed_corpus.append(filtered)

print("\n===== Preprocessed Corpus =====")
for doc in processed_corpus:
    print(doc)

# -------------------------
# 2. Create Dictionary & Corpus for LDA
# -------------------------
dictionary = corpora.Dictionary(processed_corpus)
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

# -------------------------
# 3. Apply LDA Model
# -------------------------
lda_model = models.LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    num_topics=3,   # number of topics
    passes=15       # training passes
)

print("\n===== LDA Topics =====")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

# -------------------------
# 4. Visualization
# -------------------------
print("\nSaving interactive visualization to lda_vis.html...")

vis = gensimvis.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.save_html(vis, "lda_vis.html")
print("✅ Visualization saved as lda_vis.html. Open it in your browser to view.")
