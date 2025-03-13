import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import spacy
# We use spaCy for preprocessing, but we only need the tokenizer and lemmatizer
# (for a large real-world dataset that would help with the performance)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

