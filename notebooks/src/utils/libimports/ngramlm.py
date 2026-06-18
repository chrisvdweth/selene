import numpy as np
import pandas as pd
import spacy

from collections import Counter, defaultdict

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")