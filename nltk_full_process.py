import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Sample text
text = "Monsoon intensifies across the state: The IMD has issued orange alerts in Ernakulam, Idukki, Thrissur and Kasaragod, with yellow alerts in eight other districts, warning of very heavy rainfall through mid-June"

# Tokenization
tokens = word_tokenize(text)

# Stopword list
stop_words = set(stopwords.words('english'))

# Stemming
stemmer = PorterStemmer()

# Lemmatization
lemmatizer = WordNetLemmatizer()

# Helper function to get WordNet POS
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default

# POS tagging
pos_tags = pos_tag(tokens)

# Final result
print("Word\t\t| Stopword\t| Stem\t\t| Lemma")
print("-" * 60)
for word, tag in pos_tags:
    if word.isalpha():
        is_stop = word.lower() in stop_words
        stem = stemmer.stem(word)
        lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        print(f"{word:<16} | {str(is_stop):<10} | {stem:<10} | {lemma}")
