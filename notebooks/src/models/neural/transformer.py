import math
import torch
import torch.nn as nn
import torch.nn.functional as f




class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, model_size, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, model_size)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_size, 2).float()
            * (-math.log(10000.0) / model_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Compute positional encoding
        pos_enc = self.pe[:, : x.size(1), :]
        # Add positional encoding to input vector (typicall a word embedding vector)
        x = x + pos_enc
        # Here only return the positional encoding to inspect it (not needed in practice)
        return self.dropout(x), pos_enc


    

class Attention(nn.Module):
    ### Implements Scaled Dot Product Attention
    
    def __init__(self):
        super().__init__()


    def forward(self, Q, K, V, mask=None, dropout=None):
        # All shapes: (batch_size, seq_len, hidden_size)
        
        # Perform Q*K^T (* is the dot product here)
        # We have to use torch.matmul since we work with batches!
        out = torch.matmul(Q, K.transpose(1, 2)) # => shape: (B, L, L)

        # Divide by scaling factor
        out = out / (Q.shape[-1] ** 0.5)
        
        #if mask is not None:
        #    NOT IMPLEMENTED YET
        
        # Push throught softmax layer
        out = f.softmax(out, dim=-1)
        
        # Optional: Dropout
        if dropout is not None:
            out = nn.Dropout(out, dropout)
        
        # Multiply with values V
        out = torch.matmul(out, V)
        
        return out
    

    
class AttentionHead(nn.Module):
    
    def __init__(self, model_size, qkv_size):
        super().__init__()
        self.Wq = nn.Linear(model_size, qkv_size)
        self.Wk = nn.Linear(model_size, qkv_size)
        self.Wv = nn.Linear(model_size, qkv_size)
        self.attention = Attention()
        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)
        
    def forward(self, query, key, value):
        return self.attention(self.Wq(query), self.Wk(key), self.Wv(value))
    
    

    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, model_size, num_heads):
        super().__init__()
        
        if model_size % num_heads != 0:
            raise Exception("The model size must be divisible by the number of heads!")
        
        # Define sizes of Q/K/V based on model size and number of heads
        self.qkv_size = model_size // num_heads
        
        self.heads = nn.ModuleList(
            [AttentionHead(model_size, self.qkv_size) for _ in range(num_heads)]
        )
        
        # Linear layer to "unify" all heads into one
        self.Wo = nn.Linear(model_size, model_size)
        
        # Initalize parameters of output layer
        self._init_parameters()

        
    def _init_parameters(self):
        nn.init.xavier_uniform_(self.Wo.weight)
        
        
    def forward(self, query, key, value):
        # Push Q, K, V through all the Attention Heads
        out_heads = tuple([ attention_head(query, key, value) for attention_head in self.heads ])
        
        # Concatenate the outputs of all Attention Heads
        out = torch.cat(out_heads, dim=-1)
        
        # Push concatenated outputs through last layers => output size is model_size again
        return self.Wo(out)

    
        
    
    
class FeedForward(nn.Module):
    
    def __init__(self, model_size, hidden_size=2048):
        super().__init__()
        
        # Define basic Feed Forward Network as proposed in the original paper
        self.net = nn.Sequential(
            nn.Linear(model_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, model_size),
        )

    def forward(self, X):
        return self.net(X)
    
    
    


class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, model_size, num_heads, ff_hidden_size, dropout):
        super().__init__()
               
        # MultiHeadAttention block
        self.mha1 = MultiHeadAttention(model_size, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_size)
        
        # FeedForward block
        self.ff = FeedForward(model_size, ff_hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_size)
        

    def forward(self, source):
        # MultiHeadAttentionBlock
        out1 = self.mha1(source, source, source)
        out1 = self.dropout1(out1)
        out1 = self.norm1(out1 + source)
        # FeedForward block
        out2 = self.ff(out1)
        out2 = self.dropout2(out2)
        out2 = self.norm2(out2 + out1)
        # Return final output
        return out2
    
    
    
    
class TransformerEncoder(nn.Module):
    
    def __init__(self,
                 num_layers=6, 
                 model_size=512, 
                 num_heads=8, 
                 ff_hidden_size=2048, 
                 dropout= 0.1):
        super().__init__()
        
        # Define num_layers (N) encoder layers
        self.layers = nn.ModuleList(
            [ TransformerEncoderLayer(model_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers) ]
        )

    def forward(self, source):
        for l in self.layers:
            source = l(source)
        return source

    
    
    
    
##
## Decoder
##


class TransformerDecoderLayer(nn.Module):
    
    def __init__(self, model_size, num_heads, ff_hidden_size, dropout):
        super().__init__()
        
        # 1st MultiHeadAttention block (decoder input only)
        self.mha1 = MultiHeadAttention(model_size, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_size)
        
        # 2nd MultiHeadAttention block (encoder & decoder)
        self.mha2 = MultiHeadAttention(model_size, num_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_size)
        
        self.ff = FeedForward(model_size, ff_hidden_size)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(model_size)
        
    def forward(self, target, memory):
        # 1st MultiHeadAttention block
        out1 = self.mha1(target, target, target)
        out1 = self.dropout1(out1)
        out1 = self.norm1(out1 + target)
        # 2nd MultiHeadAttention block
        out2 = self.mha2(out1, memory, memory)
        out2 = self.dropout2(out2)
        out2 = self.norm2(out2 + out1)
        # FeedForward block
        out3 = self.ff(out2)
        out3 = self.dropout3(out3)
        out3 = self.norm3(out3 + out2)
        # Return final output
        return out3
    
    
class TransformerDecoder(nn.Module):
    
    def __init__(self, 
                 num_layers=6,
                 model_size=512,
                 num_heads=8,
                 ff_hidden_size=2048,
                 dropout= 0.1):
        super().__init__()
        
        # Define num_layers (N) decoder layers
        self.layers = nn.ModuleList(
            [ TransformerDecoderLayer(model_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers) ]
        )

    def forward(self, target, memory):
        # Push through each decoder layer
        for l in self.layers:
            target = l(target, memory)
        return target
    
    
    
class Transformer(nn.Module):
    
    def __init__(self, 
                 num_encoder_layers=6, 
                 num_decoder_layers=6, 
                 model_size=512, 
                 num_heads=8, 
                 ff_hidden_size=2048, 
                 dropout= 0.1):
        super().__init__()
        
        # Definer encoder
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            model_size=model_size,
            num_heads=num_heads,
            ff_hidden_size=ff_hidden_size,
            dropout=dropout
        )
        
        #Define decoder
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            model_size=model_size,
            num_heads=num_heads,
            ff_hidden_size=ff_hidden_size,
            dropout=dropout
        )

    def forward(self, source, target):
        memory = self.encoder(source)
        return self.decoder(target, memory)