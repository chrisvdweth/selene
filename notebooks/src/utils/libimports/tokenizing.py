import re

from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import NLTKWordTokenizer, TreebankWordTokenizer, TweetTokenizer, RegexpTokenizer
from nltk import word_tokenize # Simplfied notation; it's a wrapper for the TreebankWordTokenizer

from transformers import AutoTokenizer

import spacy
nlp = spacy.load('en_core_web_sm')