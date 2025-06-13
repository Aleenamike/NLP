import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Monsoon intensifies across the state: The IMD has issued orange alerts in Ernakulam, Idukki, Thrissur and Kasaragod, with yellow alerts in eight other districts, warning of very heavy rainfall through mid-June"

# Process text
doc = nlp(text)

print("Word\t\t| Stopword\t| Lemma")
print("-" * 40)
for token in doc:
    if token.is_alpha:
        print(f"{token.text:<16} | {str(token.is_stop):<10} | {token.lemma_}")
