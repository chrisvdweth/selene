import re, collections, regex
from tqdm import tqdm




class MyWordTokenizer:
    
    def __init__(self):
        pass


    def tokenize(self, text, rules=[], exceptions=[]):

        # Split by whitespace
        tokens = text.split(' ')

        idx = 0
        while True:
            # Stop if we reach the end of the token list
            if idx >= len(tokens):
                break

            # Grab the next token
            token = tokens[idx]

            # Split the token into 2 or more subtokens; might not be necessary!
            components = self._apply_splitting_rules(token, rules, exceptions)

            # Just for safety, we ignore any empty component
            # (this might happen during the split and it's more convenient to handle it here)
            components = [ c.strip() for c in components if len(c.strip()) > 0]

            # Check if the current token was indeed split into 2 more components
            if len(components) > 1:
                # If the token was split create a new token list
                tokens = tokens[:idx] + components + tokens[idx+1:]
            else:
                # If the token was NOT split; just go to the next token in the list
                idx +=1

        # Return final list of tokens
        return tokens

    
    def _apply_splitting_rules(self, token, rules, exceptions):

        for exception in exceptions:
            if exception(token) == True:
                return [token]

        for rule in rules:
            rule_applied, components = rule(token)
            if rule_applied == True:
                return components

        # If we didn't find any reason to split, return the token as the only subtoken itself
        return [token]









class MyBpeTokenizer():

    PRE_TOKENIZE__SPLIT = 0
    PRE_TOKENIZE__GPT2  = 1
    
    def __init__(self, eos='Ġ', pretokenize=PRE_TOKENIZE__SPLIT):
        self._pretokenize = pretokenize
        self._eos = eos
        self._vocabulary = {}
        self._corpus_state = {}
        self._merges = []


    def _init(self, docs: list):
        # Initialize vocabulary, corpus state, and list of merges
        self._vocabulary = set()
        self._corpus_state = collections.defaultdict(int)
        self._merges = []
        # Loop over all documents
        for doc in docs:
            # Add all characters in the current document to the vocabulary
            self._vocabulary.update(set(doc))
            # For each word in the document, generate the sequence and update add it to the corpus state
            for word in self._pretokenize_text(doc):
                self._corpus_state[self._generate_sequence(word)] += 1
        # Remove whitespace character from final vocabulary
        self._vocabulary.discard(" ")
        # Add EOS token
        self._vocabulary.add(self._eos)
        self._corpus_state = dict(self._corpus_state)

    
    def _pretokenize_text(self, text):
        if self._pretokenize == MyBpeTokenizer.PRE_TOKENIZE__SPLIT:
            return text.split()
        elif self._pretokenize == MyBpeTokenizer.PRE_TOKENIZE__GPT2:
            gpt2pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+""")
            return regex.findall(gpt2pattern, text)
        else:
            raise Exception("Unknown pretokenization method.")
        
    
    def _generate_sequence(self, word):
        return ' '.join([self._eos] + (list(word)))


    def _find_most_frequent_token_pair(self):
        token_pair_counts = collections.defaultdict(int)
        # Loop over corpus state to sum up occurrences of each existing token pair
        for word, freq in self._corpus_state.items():
            sequence = word.split()
            for i in range(len(sequence)-1):
                token_pair_counts[f"{sequence[i]} {sequence[i+1]}"] += freq
        # Return the most frequent pair (if their are ties, we just randomly break them)
        return max(token_pair_counts.keys(), key=(lambda key: token_pair_counts[key]))


    def _perform_merge(self, token_pair):
        # Create new token by merging token pair
        new_token = re.sub(' ', '', token_pair)
        # Create merge as tuple of token pair and new token
        merge = (token_pair, new_token)
        # Add new token to vocabulary
        self._vocabulary.add(new_token)
        # Define search pattern
        pattern = re.compile(r"(?<!\S)" + re.escape(token_pair) + r"(?!\S)")
        # Loop through corpus state and record which keys/sequences need to be updated
        translate = {}
        for sequence, count in self._corpus_state.items():
            for match in pattern.finditer(sequence):
                translate[sequence] = pattern.sub(new_token, sequence)
        # Perform the update of keys/sequences
        for old, new in translate.items():
            self._corpus_state[new] = self._corpus_state.pop(old)
        return merge
        
    
    def fit(self, docs, max_vocab_size=100, verbose=False):
        # Initialize the corpus state and vocabulary
        if verbose == True:
            print("Initilize corpus and vocabulary...")
        self._init(docs)
        # Calculate the number of merging steps to be performed
        num_iter = max(0, (max_vocab_size-len(self._vocabulary)))
        # Perform the required number of merging steps; might stop sooner if not merge possible
        if verbose == True:
            print(f"Perform {num_iter} iterations...")
        for _ in tqdm(range(num_iter)):
            # Find the most frequent pair (if their are ties, we just randomly break them)
            try:
                top_token_pair = self._find_most_frequent_token_pair()
            except:
                break
            # Update corpus state and the vocabulary
            merge = self._perform_merge(top_token_pair)
            # Add newly merged symbol to vocabulary
            self._merges.append(merge)
        return self


    def tokenize(self, doc: str):
        tokens = []
        for word in self._pretokenize_text(doc):
            tokens.extend(self._tokenize_word(word))
        return tokens

    
    def _tokenize_word(self, word):
        #
        sequence = self._generate_sequence(word)
        #
        for p, m in self._merges:
            if p not in sequence:
                continue
            pattern = re.compile(r'(?<!\S)' + re.escape(p) + r'(?!\S)')
            sequence = pattern.sub(m, sequence)
        #   
        return sequence.split(' ')


    def detokenize(self, tokens: list):
        doc = ''.join(tokens)
        return re.sub(self._eos, " ", doc).strip()



class MyWordPieceTokenizer():

    PRE_TOKENIZE__SPLIT = 0
    PRE_TOKENIZE__GPT2  = 1

    def __init__(self, ctoken='##', pretokenize=PRE_TOKENIZE__SPLIT):
        self._pretokenize = pretokenize
        self._ctoken = ctoken
        self._vocabulary = {}
        self._corpus_state = {}
        self._merges = []

    
    def _init(self, docs: list):
        # Initialize vocabulary, corpus state, and list of merges
        self._vocabulary = set()
        self._corpus_state = collections.defaultdict(int)
        self._merges = []
        # Loop over all documents
        for doc in docs:
            # For each word in the document, generate the sequence and update add it to the corpus state
            for word in self._pretokenize_text(doc):
                # Update vocabulary
                for idx, char in enumerate(word):
                    if idx == 0:
                        self._vocabulary.add(char)
                    else:
                        self._vocabulary.add(f"{self._ctoken}{char}")
                # Update sequence count
                self._corpus_state[self._generate_sequence(word)] += 1

    
    def _pretokenize_text(self, text):
        if self._pretokenize == MyWordPieceTokenizer.PRE_TOKENIZE__SPLIT:
            return text.split()
        elif self._pretokenize == MyWordPieceTokenizer.PRE_TOKENIZE__GPT2:
            gpt2pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+""")
            return regex.findall(gpt2pattern, text)
        else:
            raise Exception("Unknown pretokenization method.")

    
    def _generate_sequence(self, word):
        return ' '.join([c if i == 0 else f"{self._ctoken}{c}" for i, c in enumerate(word)])


    def _find_best_token_pair(self):
        token_counts = collections.defaultdict(int)
        token_pair_counts = collections.defaultdict(int)
        
        for word, freq in self._corpus_state.items():
            sequence = word.split()
            if len(sequence) == 1:
                token_counts[sequence[0]] += freq
                continue
            for i in range(len(sequence)-1):
                pair = (sequence[i], sequence[i+1])
                token_counts[f"{sequence[i]}"] += freq
                token_pair_counts[pair] += freq
            token_counts[sequence[-1]] += freq
    
        token_pair_scores = { 
            ' '.join(pair): count / (token_counts[pair[0]] * token_counts[pair[1]]) 
            for pair, count in token_pair_counts.items() 
        }
    
        # Return the most frequent pair (if their are ties, we just randomly break them)
        return max(token_pair_scores.keys(), key=(lambda key: token_pair_scores[key]))


    def _create_new_token(self, token_pair):
        t1, t2 = token_pair.split()
        return ''.join([t1, re.sub(self._ctoken, "", t2)])
    

    def _perform_merge(self, token_pair):
        # Create new token by merging token pair
        new_token = self._create_new_token(token_pair)
        # Create merge as tuple of token pair and new token
        merge = (token_pair, new_token)
        # Add new token to vocabulary
        self._vocabulary.add(new_token)
        # Define search pattern
        pattern = re.compile(r"(?<!\S)" + re.escape(token_pair) + r"(?!\S)")
        # Loop through corpus state and record which keys/sequences need to be updated
        matches = {}
        for sequence, count in self._corpus_state.items():
            for match in pattern.finditer(sequence):
                matches[sequence] = pattern.sub(new_token, sequence)
        # Perform the update of keys/sequences
        for old, new in matches.items():
            self._corpus_state[new] = self._corpus_state.pop(old)
        # Return the merge
        return merge


    def fit(self, docs, max_vocab_size=100, verbose=False):
        # Initialize the corpus state and vocabulary
        if verbose == True:
            print("Initilize corpus and vocabulary...")
        self._init(docs)
        # Calculate the number of merging steps to be performed
        num_iter = max(0, (max_vocab_size-len(self._vocabulary)))
        # Perform the required number of merging steps; might stop sooner if not merge possible
        if verbose == True:
            print(f"Perform {num_iter} iterations...")
        for _ in tqdm(range(num_iter)):
            # Find the most frequent pair (if their are ties, we just randomly break them)
            try:
                top_token_pair = self._find_best_token_pair()
            except:
                break
            # Update corpus state and the vocabulary
            merge = self._perform_merge(top_token_pair)
            # Add newly merged symbol to vocabulary
            self._merges.append(merge)
        return self


    def tokenize(self, doc: str):
        pretokens = self._pretokenize_text(doc)
        tokens = []
        for pt in pretokens:
            tokens.extend(self._tokenize_word(pt))
        return tokens
    
    
    def _tokenize_word(self, word):
        sequence = self._generate_sequence(word)
        for p, m in self._merges:
            if p not in sequence:
                continue
            p = re.compile(r'(?<!\S)' + re.escape(p) + r'(?!\S)')
            sequence = p.sub(m, sequence)
        return sequence.split(' ')

    
    def detokenize(self, tokens: list):
        doc = ' '.join(tokens)
        return re.sub(f" {self._ctoken}", "", doc).strip()