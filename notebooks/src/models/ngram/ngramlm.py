import numpy as np
import pandas as pd
import spacy

from tqdm import tqdm
from collections import Counter, defaultdict
from itertools import product

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


class NGramLanguageModel:
    def __init__(self, N=2, k=1.0, lowercase=False):
        self.lowercase = lowercase
        if N < 2:
            raise ValueError("N must be > 1")
        self.N = N
        self.k = k
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()
        self.SOS = "[SOS]"
        self.EOS = "[EOS]"
        # Dictionary that keeps list of candidate words given context
        # When generating a text, we only pick from those candidate words
        self.contexts = {}
        self.token_count = 0

    def reset(self):
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()
        self.contexts = {}
        self.token_count = 0        
    
    def _preprocess_sentence(self, sentence):
        doc = nlp.make_doc(sentence)
        # Tokenize sentence using spaCy
        if self.lowercase is True:
            tokens = [ t.text.lower() for t in doc ]
        else:
            tokens = [ t.text for t in doc ]
        # Add special tokens and return final list
        return [self.SOS]*(self.N-1) + tokens + [self.EOS]
    
    def train(self, sentences: list[str]):
        self.reset();
        # Iterate through all sentences in dataset
        for sentence in tqdm(sentences):
            # Tokenize sentence and add special tokens
            tokens = self._preprocess_sentence(sentence)
            # Add all tokens to the vocabulary
            self.vocab.update(tokens)
            # Update token count
            self.token_count += len(tokens)
            # Iterate through all tokens to generate ngrams and contexts
            for i in range(self.N-1, len(tokens)):
                # Get ngram as context+word
                context = tuple(tokens[i - self.N + 1:i])
                word = tokens[i]
                # Update ngram and context counters
                self.ngram_counts[context + (word,)] += 1
                self.context_counts[context] += 1
                # Update list of possible words for context
                if context in self.contexts:
                    self.contexts[context].add(word)
                else:
                    self.contexts[context] = set([word])

    def compute_ngram_probability(self, word, context):
        # Shorten context if needed
        context = tuple(context[-(self.N-1):]) if self.N > 1 else tuple()
        # Numerator: ngram count + smoothing
        numerator = self.ngram_counts[context + (word,)] + self.k
        # Denominator: context count + smoothing
        denominator = self.context_counts[context] + self.k * len(self.vocab)
        # Return probability as the relative frequency (smoothed)
        return numerator / denominator

    def compute_sentence_log_probability(self, sentence):
        # Tokenize sentence and add special tokens
        tokens = self._preprocess_sentence(sentence)
        # Intialize log probability to 0
        log_prob = 0.0
        # Iterate through all tokens to generate ngrams and contexts
        for i in range(self.N-1, len(tokens)):
            # Get ngram as context+word
            context = tuple(tokens[i-self.N+1:i])
            word = tokens[i]
            # Compute the log probability of ngram and add to final result
            log_prob += np.log(self.compute_ngram_probability(word, context))
        # Return final log probability
        return log_prob

    def perplexity(self, sentences):
        total_log_prob = 0.0
        total_tokens = 0

        for sentence in sentences:
            tokens = self._preprocess_sentence(sentence)
            total_log_prob += self.compute_sentence_log_probability(sentence)
            total_tokens += len(tokens)

        return np.exp(-total_log_prob / total_tokens)

    def generate(self, max_tokens: int = 30, start_context: list[str] = [], topk: int = 20):
        # The following block merely prepares the first context; note that the context is always of size
        # (self.n - 1) so depending on the start_context (representing the start/seed words), we need to
        # pad or cut off the start_context.
        if len(start_context) == (self.N-1):
            context = start_context.copy()
        elif len(start_context) < (self.N-1):
            context = ((self.N - (len(start_context)+1)) * [self.SOS]) + start_context.copy()
        elif len(start_context) > (self.N-1):
            context = start_context[-(self.N-1):].copy()
        output = start_context.copy()  

        # Generate the next word in each iteration
        for _ in range(max_tokens):
            # Compute the probabilities for all word
            words = np.asarray(list(self.contexts[tuple(context)]))
            probs = np.asarray([self.compute_ngram_probability(w, context) for w in words])
            # If specified, limit choice of word to topk words with the highest probabilities
            if topk is not None:
                # Make sure that we have words to select from
                topk = min(topk, len(words))
                # Get indices of the k largest elements (unsorted)
                top_probs_indices = np.argpartition(probs, -topk)[-topk:]
                # Extrat the relevant words and probabilities
                words = words[top_probs_indices]
                probs = probs[top_probs_indices]
                # Normalize probabilities to they sum up to 1
                probs = probs / probs.sum()
            # Randomly select next word (proportional to probabilities)
            word = np.random.choice(words, size=1, p=probs)[0]
            # If the next word is the EOS token, stop generating tokens
            if word == self.EOS:
                break
            # Add word to output
            output.append(word)
            # Update the context by added the new word and dropping the first worf
            context = context[1:] + [word]
        # Return joined tokens as the final output
        return " ".join(output)






def generate_bigram_data(model, tokens, mode='counts', decimals=4):
    n = len(tokens)
    pairs = np.asarray(list(product(tokens, repeat=2))).reshape(n, n, 2)

    data_unigrams = np.zeros((1, n))
    for idx, unigram in enumerate(tokens):
        if mode.lower() == 'counts':
            data_unigrams[0,idx] = int(model.context_counts[(unigram, )])
        if mode.lower() == 'probs':
            data_unigrams[0,idx] = np.round(model.context_counts[(unigram, )] / model.token_count, decimals=decimals)
    
    data_bigrams  = np.zeros((n, n))   
    for row_idx, row in enumerate(pairs):
        for col_idx, bigram in enumerate(row):
            bigram = tuple(bigram)
            if mode.lower() == 'counts':
                data_bigrams[row_idx,col_idx] = int(model.ngram_counts[bigram])
            elif mode.lower() == 'probs':
                data_bigrams[row_idx,col_idx] = np.round(model.compute_ngram_probability(bigram[1], (bigram[0],)), decimals=decimals)

    if mode == 'counts':
        data_bigrams  = data_bigrams.astype(int)
        data_unigrams = data_unigrams.astype(int)
    
    df_bigrams  = pd.DataFrame(data_bigrams,  columns=tokens, index=tokens)
    df_unigrams = pd.DataFrame(data_unigrams, columns=tokens, index=[mode])

    return df_bigrams, df_unigrams