import torch
from torch import nn, optim
from torch.utils.data import Dataset
import numpy as np


############## Task2- Implement Transformer ##################################
class TransformerDataset(Dataset):
    def __init__(self, encoder_inputs, decoder_inputs, decoder_targets):
        super(TransformerDataset,self).__init__()
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets
    
    def __len__(self):
        return self.encoder_inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.encoder_inputs[idx], self.decoder_inputs[idx], self.decoder_targets[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.embedding_size, self.max_length = embedding_size, max_length
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', self.get_pe())
        
    def get_pe(self):
        pe = torch.zeros((self.max_length, self.embedding_size))
        position = torch.arange(0, self.max_length, dtype=torch.float32).reshape(-1, 1)
        div = torch.pow(10000, torch.arange(0, self.embedding_size, 2, dtype=torch.float32)/300)
        pe[:, 0::2] = torch.sin(position/div)
        pe[:, 1::2] = torch.cos(position/div)
        pe = pe.unsqueeze(0)  # add batch dimension
        return pe
    
    def forward(self, X):
        '''
        X: [batch_size, seq_len, d_model]
        '''
        X = X + self.pe[:, :X.size(1), :]
        return self.dropout(X)
    
    
def pad_mask(X, dec_X= None):
    """
    X: [batch_size, length]
    """
    batch_size, length_enc = X.size()
    if dec_X is None:
        batch_size, length_dec = X.size()
    else:
        batch_size, length_dec = dec_X.size()
    # X.data.eq[0]return a matrix same with X; filled by True or False
    # if value in X is 0, fill with True, else False
    pad_mask = X.data.eq(0)
    # expand to size [batch_size, length_dec, length_enc]
    pad_mask = pad_mask.unsqueeze(1).expand(batch_size, length_dec, length_enc)
    return pad_mask


def subsequence_mask(seq):
    """
    Mask future words
    seq: [batch_size, length]
    
    return: [batch_size, length, length] 
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # Upper triangular matrix
    sub_seq_mask = np.triu(np.ones(attn_shape), k=1)
    sub_seq_mask = torch.from_numpy(sub_seq_mask).byte()
    return sub_seq_mask
    
    
class ScaledDotProductAttention(nn.Module):
    '''
    This module dive to calcuate Q, K, V and return the context
    '''
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        
    def forward(self, Q, K, V, attn_mask):
        '''
        For self-attention:
            Q = K = [batch_size, n_heads, length, d_k]
            V = [batch_size, n_heads, length, d_v]
            attn_mask = [batch_size, n_heads, length, length]
        
        For decoder-encoder-attention:
            Q = [batch_size, n_heads, length_decoder, d_k]
            K = [batch_size, n_heads, length_encoder, d_k]
            V = [batch_size, n_heads, length_encoder, d_v]
            attn_mask = [batch_size, n_heads, length_decoder, length_encoder]
        '''
        # Calculate Q, K; divide sqrt(d_k) for normalization scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(3))
        # Add -inf to masked value, then it will be zero in softmax 
        scores.masked_fill_(attn_mask, -1e9)
        attn_weight = nn.Softmax(dim=-1)(scores)
        
        # use attn_weight extract context
        context = torch.matmul(attn_weight, V)
        return context
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, k_dim, v_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads, self.k_dim, self.v_dim = n_heads, k_dim, v_dim
        self.W_Q = nn.Linear(embedding_size, k_dim * n_heads, bias = False)
        self.W_K = nn.Linear(embedding_size, k_dim * n_heads, bias = False)
        self.W_V = nn.Linear(embedding_size, v_dim * n_heads, bias = False)
        self.fc = nn.Linear(n_heads * v_dim, embedding_size, bias = False)
        
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        For self-attention:
            input_Q, input_K, input_V = X = [batch_size, length, embedding]
            attn_mask = [batch_size, length, length]
        
        For decoder-encoder-attention:
            input_Q = decoder_outputs = [batch_size, length_decoder, embedding]
            input_K, input_V = encoder_outputs = [batch_size, length_encoder, embedding]
            attn_mask = [batch_size, length_decoder, length_encoder]
        '''
        n_heads, k_dim, v_dim = self.n_heads, self.k_dim, self.v_dim
        residual, batch_size = input_Q, input_Q.size(0)
        # Q = X * W_Q 
        # Convert big matrix Q into multi-head matrix
        # Q: [batch_size, n_heads, length, k_dim]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, k_dim).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, k_dim).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, v_dim).transpose(1, 2)
        
        # Calculate attention and Extract context
        # attn_mask: [batch_size, n_heads, length, length]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        # context: [batch_size, n_heads, length, v_dim]
        context = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # context: [batch_size, length, n_heads * v_dim]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * v_dim)
        output = self.fc(context)
        
        # Add residual and Layer Normalization
        output = nn.LayerNorm(output.size(2)).to(output.device)(output + residual)
        return output
        
        
class FeedForwardNet(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(FeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, hidden_size, bias= False),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size, bias = False)
        )
        
    def forward(self, inputs):
        '''
        inputs: [batch_size, length, embedding]
        '''
        residual = inputs
        # output:  [batch_size, length, embedding]
        output = self.fc(inputs)
        # Add residual and Layer Normalization
        output = nn.LayerNorm(output.size(2)).to(output.device)(output + residual)
        return output
    
    
class EncoderLayer(nn.Module):
    def __init__(self, embedding_size, k_dim, v_dim, n_heads, hidden_size):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_size, k_dim, v_dim, n_heads)
        self.feed_forward = FeedForwardNet(embedding_size, hidden_size)
        
    def forward(self, X, mask):
        '''
        X: [batch_size, length, embedding]
        mask: [batch_size, length, embedding]
        '''
        output = self.self_attention(X, X, X, mask)
        # output.shape == X.shape
        output = self.feed_forward(output)
        return output
    
    
class DecoderLayer(nn.Module):
    def __init__(self, embedding_size, k_dim, v_dim, n_heads, hidden_size):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_size, k_dim, v_dim, n_heads)
        self.decoder_encoder_attention = MultiHeadAttention(embedding_size, k_dim, v_dim, n_heads)
        self.feed_forward = FeedForwardNet(embedding_size, hidden_size)

    def forward(self, decoder_inputs, encoder_outputs, self_attention_mask, decoder_encoder_attention_mask):
        '''
        decoder_inputs: [batch_size, decoder_length, embedding_size]
        encoder_outputs: [batch_size, encoder_length, embedding_size]
        self_attention_mask: [batch_size, decoder_length, decoder_length]
        decoder_encoder_attention_mask: [batch_size, decoder_length, encoder_length]
        '''
        # Decoder-Self-Attention
        outputs = self.self_attention(
            decoder_inputs, decoder_inputs, decoder_inputs, self_attention_mask
        )
        # Decoder-Encoder-Attention
        outputs= self.decoder_encoder_attention(
            outputs, encoder_outputs, encoder_outputs, decoder_encoder_attention_mask
        )
        # FeedForward Layer
        outputs = self.feed_forward(outputs)
        return outputs
    
    
    
############## Task3- Implement BERT ##################################
    
    
    
    
    