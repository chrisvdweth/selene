import re, json
import pandas as pd
from unidecode import unidecode

import spacy
try:
	nlp = spacy.load("en_core_web_sm")
except OSError:
	# Fallback keeps notebooks runnable when the small English model is not installed.
	nlp = spacy.blank("en")