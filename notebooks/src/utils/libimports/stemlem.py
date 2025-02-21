import string
import nltk
import json

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize
from nltk import pos_tag

import spacy
nlp = spacy.load('en_core_web_sm')