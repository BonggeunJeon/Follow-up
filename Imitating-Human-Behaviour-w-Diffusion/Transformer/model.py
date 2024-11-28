import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    """ 
        The Multi-Head Attention sublayer
    """
    
    def __init__(self, d_model, n_heads, d_queries, d_values, dropout, in_decoder=False):
        """ 
        :param d_model : size of vectors throughout the transformer model, i.e. input and output sizes for this sublayer
        :param n_heads : number of heads in the multi-head attention
        :param d_queries : size of query vectors (and also the size of the key vectors)
        :param d_values : size of value vectors
        :param dropout : dropout probability 
        :param in_decoder : is this Multi-Head Attention sublayer instance in the decoder?
        """
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_keys = d_queries # size of key vector, same as of the query vectors to allow dot-products for similarity
        
        self.in_decoder = in_decoder
        
        # A linear projection to cast (n_heads sets of) queries from the input query sequences
        self.cast_queries =nn.Linear(d_model, n_heads * d_queries) # in_features : d_model, output_features : n_heads * d_queries
        
        # A linear projection to cast (n_heads sets of) keys and values from the input reference sequences
        self.cast_keys_values = nn.Linear(d_model, n_heads * (d_queries + d_values))
        
        # A linear projection to cast (n_heads sets of) computed attention-weighted vectors to output vectors (of the same size as input query vectors)
        self.cast_output = nn.Linear(n_heads * d_values, d_model)
        
        # Softmax Layer
        self.softmax = nn.Softmax(dim=-1) # dim=-1 뒤에서부터 연산을 하나?
        
        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)
        
    def forward(self, query_sequences, key_value_sequences, key_value_sequence_lenghts):
        """ 
            Forward prpo.
            
            :param query_sequences : the input query sequence, a tensor of size (N, query_sequence_pad_lengh, d_model)
            :param key_value_sequences : the sequences to be queried against, a tensor of size (N, key_value_sequence_pad_length, d_model)
            :param key_value_sequences_lengths : true lengths of the key_value_sequences, to be able to ignore pads, a tensor of size (N)
            :return : attention-weighted output sequences for the query sequences, a tensor of size (N, query_sequence_pad_lengh, d_model)
        """
        
        batch_size = query_sequences.size(0) # batch size (N) in number of sequences
        query_sequences_pad_length = query_sequences.size(1)
        key_value_sequence_pad_lenght = key_value_sequences.size(1)
        
        # Is this self-attention?
        self_attention = torch.equal(key_value_sequences, query_sequences)
        
        # Store input for adding layer
        input_to_add = query_sequences.clone()
        
        # Apply layer normalization
        query_sequences = self.layer_norm(query_sequences) # (N, query_sequence_pad_length, d_model)
        if self_attention:
            key_value_sequences = self.layer_norm(key_value_sequences) # (N, key_value_sequence_pad_langth, d_model)
        
        # Project input sequences to queries, keys, values
        queries = self.cast_queries(query_sequences) 
        keys, values = self.cast_keys_values(key_value_sequences).split(split_size=self.n_heads * self.d_keys, dim=-1)
        
        # Split the last dimension by the n_heads subspaces
        queries = queries.contiguous().view(batch_size, query_sequences_pad_length, self.n_heads, self.d_queries)
        keys = keys.contiguous().view(batch_size, key_value_sequence_pad_lenght, self.n_heads, self.d_keys)
        values = values.contiguous().view(batch_size, key_value_sequence_pad_lenght, self.n_heads, self.d_values)
        
        # Re-arrange axes such that the last two dimensions are the sequence lenghts and the queries/key/values
        # And then, for convenience, convert to 3D tensors by merging the batch and n_heads dimensions
        # This is to prepare it for the batch matrix multiplication (i.e. the dot product)
        queries = queries.permute(0, 2, 1, 3).contiguous().view(-1, query_sequences_pad_length, self.d_queries)
        keys = keys.permute(0, 2, 1, 3).contiguous().view(-1, key_value_sequence_pad_lenght, self.d_keys)
        values = values.permute(0, 2, 1, 3).contiguous().view(-1, key_value_sequence_pad_lenght, self.d_values)
        
        # Perform multi-head attention
        
        # Perform dot-products
        attention_weights = torch.bmm(queries, keys.permute(0, 2, 1))
        
        # Scale dot-products
        attention_weights = (1. / math.sqrt(self.d_keys)) * attention_weights
        
        # Before computing softmax weights, prevent queires from attending to certain keys
        
        # Mask 1 : keys that are pads
        not_pad_in_keys = torch.LongTensor(range(key_value_sequence_pad_lenght)).unsqueeze(0).unsqueeze(0).expand_as(
            attention_weights).to(device) 
        
        not_pad_in_keys = not_pad_in_keys < key_value_sequence_lenghts.repeat_interleave(self.n_heads).unsqueeze(1).unsqueeze(2).expand_as(
            attention_weights)
        
        # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        attention_weights = attention_weights.masked_fill(~not_pad_in_keys, -float('inf'))
        
        # Mask 2 : if this is self-attention in the decoder, keys chronologically ahead of queries
        if self.in_decoder and self_attention:
            # Therefore, a position [n, i, j] is valid only if j <= i
            not_future_mask = torch.ones.like(attention_weights).tril().bool().to(device)
            
            attention_weights = attention_weights.masked_fill(~not_future_mask, -float('inf'))
            
        # Compute softmax along the key dimension / Softmax의 특성이 뭐야?
        """ 
            Softmax function normalizes output values from 0 to 1. And total value have to be 1.
        """
        attention_weights = self.softmax(attention_weights)
        
        # Apply droput 
        attention_weights = self.apply_dropout(attention_weights)
        
        # Calculate sequences as the weighted sums of values based on these softmax weights
        sequences = torch.bmm(attention_weights, values) 
        
        # Unmerge batch and n_heads dimensions and restore original order of axes / contiguous 성질이 뭐야?
        sequences - sequences.contiguous().view(batch_size, self.n_heads, query_sequences_pad_length, self.d_values).permute(0, 2, 1, 3)
        
        # Concatenate the n_heads subspace 
        sequences = sequences.contiguous().view(batch_size, query_sequences_pad_length, -1)
        
        # Transform the concatenated subspace-sequences into a single output of size d_model
        sequences = self.cast_output(sequences)
        
        # Apply dropout and residual connection
        sequences = self.apply_dropout(sequences) + input_to_add 
        
        return sequences
    
class PositionWiseFCNetwork(nn.Module):
    """ 
        The Position-Wise Feed Forward Network sublayer.
        
        Position 단위로 Feed Forward Network랑 연결시켜준다는 것인가?
        
        I think, it means that position can be connected feed forward network
    """
    def __init__(self, d_model, d_inner, dropout):
        """ 
        :param d_moel : size of vector throughout the transformal model, i.e. input and output sizes for this sublayer
        :param d_inner : an intermediate size
        :param drouput : dropout probability 
        """
        super(PositionWiseFCNetwork, self).__init__()
        
        self.d_model = d_model
        self.d_inner = d_inner
        
        # Layer-norm layer - Is it the layer for normalization? 
        self.layer_norm = nn.LayerNorm(d_model)
        
        # A linear layer to project from the input size to an intermediate size
        self.fc1 = nn.Linear(d_model, d_inner)
        
        # ReLU
        """
            Relu = max(0 ,x)
            
            1. The sigmoid function contains the exponentional function that can make calculation slower than ReLU.
            2. In Sigmoid function, if we have extream low or high values, the output will be converged into 0 or 1 which makes training too slow. 
            3. But ReLU function is simple with respect to calculation and can make training fast.
        """
        self.relu = nn.ReLU()
        
        # A linear layer to project from the intermediate size to the output size (same as the input size)
        self.fc2 = nn.Linear(d_inner, d_model)
        
        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)
        
    def forward(self, sequences):
        """ 
            Forward prop.
            
            :param sequences : input sequences, a tensor of size (N, pad_length, d_model)
            :return : transformed output sequences, a tensor of size (N, pad_length, d_model)
        """
        # Store input for adding layer
        input_to_add = sequences.clone()
        
        # Apply layer-norm
        sequences  = self.layer_norm(sequences) 
        
        # Transform position-wise
        sequences = self.apply_dropout(self.relu(self.fc1(sequences)))
        sequences = self.fc2(sequences)
        
        # Apply dropout and residual connection
        sequences = self.apply_dropout(sequences) + input_to_add
        
        return sequences
    
        
        