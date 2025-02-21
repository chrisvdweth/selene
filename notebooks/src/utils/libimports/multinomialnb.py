import numpy as np
import pandas as pd
import json

from collections import defaultdict

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer