import numpy as np
from tqdm import tqdm

class Vocabulary:

    def __init__(self, tokens, special_tokens: list=[]):
        self.token2index = { token:index for index, token in enumerate(list(special_tokens)+list(tokens)) }
        self.index2token = { index:token for token, index in self.token2index.items() }
        self.default_index = -1

    def set_default_index(self, default_index: int):
        if default_index not in self.index2token: 
            raise Exception("Invalid default index.")
        self.default_index = default_index

    def encode(self, tokens: list[str]):
        return np.asarray([ self.token2index[t] if t in self.token2index else self.default_index for t in tokens ])

    def decode(self, indices: list[int], default_token="[???]"):
        return [ self.index2token[i] if i in self.index2token else default_token for i in indices ]
    
    def __getitem__(self, token: str):
        return self.token2index[token]

    def __len__(self):
        return len(self.index2token)


class Vectors:

    def __init__(self, vocab):
        self.vocab = vocab

    def create_embedding_matrix(self, embed_file_name, embed_dim, sep=' '):
        # Read number of lines for progress bar (acceptable overhead?)
        num_vectors = sum(1 for i in open(embed_file_name, 'rb'))
        # Initialize embeddings matrix to handle unknown tokens
        embed_matrix = np.random.rand(len(self.vocab.index2token), embed_dim)
        # Iterate over file and add each relevant word vector to matrix
        with open(embed_file_name) as file:
            for idx, line in tqdm(enumerate(file), total=num_vectors):
                elems = line.split(sep)
                token = elems[0]
                try:
                    token_index = self.vocab.token2index[token]
                    embed_matrix[token_index] = elems[1:]
                except:
                    pass                
        # Return embedding matrix
        return embed_matrix    