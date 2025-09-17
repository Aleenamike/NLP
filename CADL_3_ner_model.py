# -------------------------------------------------------------
# Named Entity Recognition (NER) with spaCy
# Extract structured info: Persons and Organizations
# -------------------------------------------------------------

# Install if needed:
# pip install spacy pandas
# python -m spacy download en_core_web_sm

import spacy
import pandas as pd

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Example dataset: Job postings
# -------------------------
corpus = [
    "Google is hiring Data Scientists in Bangalore. Contact John Doe for details.",
    "Microsoft announced new AI research positions in Hyderabad, reach out to Jane Smith.",
    "Amazon is looking for Software Engineers in Seattle.",
    "Infosys offered a Project Manager role in Pune, supervised by Rahul Kumar."
]

print("===== Original Corpus =====")
for doc in corpus:
    print(doc)

# -------------------------
# 1. Apply NER with spaCy
# -------------------------
print("\n===== Named Entities Identified =====")
all_entities = []

for text in corpus:
    doc = nlp(text)
    for ent in doc.ents:
        print(f"{ent.text} -> {ent.label_}")
        all_entities.append((text, ent.text, ent.label_))

# -------------------------
# 2. Create Structured Table
# -------------------------
df = pd.DataFrame(all_entities, columns=["Sentence", "Entity", "Label"])

# Filter only Person & Organization
structured_df = df[df["Label"].isin(["PERSON", "ORG"])]

print("\n===== Structured Information (Person & Organization) =====")
print(structured_df)
