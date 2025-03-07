import pickle
from tqdm import tqdm
from collections import Counter

# We use spaCy for preprocessing, but we only need the tokenizer and lemmatizer
# (for a large real-world dataset that would help with the performance)
import spacy
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])