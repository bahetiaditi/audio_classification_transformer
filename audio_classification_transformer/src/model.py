import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SoundClassifier(nn.Module):
    def __init__(self, output_size, sequence_length):
        super(SoundClassifier, self).__init__()

        self.conv_layer1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_layer2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_layer3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)


        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)


        feature_dim = sequence_length //(2**3)  # 3 pooling layers that halve the input size each time

        self.dense_layer1 = nn.Linear(64 * feature_dim, 128)
        self.dense_layer2 = nn.Linear(128, output_size)

    def forward(self, input_tensor):
        input_tensor = self.max_pool(F.relu(self.conv_layer1(input_tensor)))
        input_tensor = self.max_pool(F.relu(self.conv_layer2(input_tensor)))
        input_tensor = self.max_pool(F.relu(self.conv_layer3(input_tensor)))

        input_tensor = input_tensor.view(input_tensor.size(0), -1)

        input_tensor = F.relu(self.dense_layer1(input_tensor))
        output_tensor = self.dense_layer2(input_tensor)

        return output_tensor

class SoundClassifierupdated(nn.Module):
    def __init__(self, output_size, sequence_length):
        super(SoundClassifierupdated, self).__init__()

        self.conv_layer1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_layer2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_layer3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        feature_dim = sequence_length // 8

        self.dense_layer1 = nn.Linear(64 * feature_dim, 128)
        self.dense_layer2 = nn.Linear(128, output_size)

        self.layer_dropout = nn.Dropout(p=0.5)

    def forward(self, input_tensor):
        input_tensor = self.max_pool(F.relu(self.conv_layer1(input_tensor)))
        input_tensor = self.max_pool(F.relu(self.conv_layer2(input_tensor)))
        input_tensor = self.max_pool(F.relu(self.conv_layer3(input_tensor)))

        input_tensor = input_tensor.view(input_tensor.size(0), -1)

        input_tensor = F.relu(self.dense_layer1(input_tensor))
        input_tensor = self.layer_dropout(input_tensor)
        output_tensor = self.dense_layer2(input_tensor)

        return output_tensor

class AudioFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, sequence_length=16000):
        super(AudioFeatureExtractor, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        ])
        self.pooling_layer = nn.MaxPool1d(kernel_size=2, stride=2)

        self.feature_dim = self.calculate_feature_dim(sequence_length)

    def calculate_feature_dim(self, length):
       with torch.no_grad():
         sample_input = torch.autograd.Variable(torch.rand(1, 1, length))
         features = self.extract_features(sample_input)
         total_size = features.data.view(1, -1).size(1)
       return total_size

    def extract_features(self, inputs):
        for conv in self.convs:
            inputs = self.pooling_layer(F.relu(conv(inputs)))
        return inputs

    def forward(self, inputs):
        features = self.extract_features(inputs)
        return features.view(-1, self.feature_dim)

def compute_scaled_dot_product_attention(Q, K, V):
    """
    Computes the scaled dot-product attention.

    Args:
    - Q (torch.Tensor): Queries tensor of shape (..., seq_len, depth)
    - K (torch.Tensor): Keys tensor of shape (..., seq_len, depth)
    - V (torch.Tensor): Values tensor of shape (..., seq_len, depth_v)

    Returns:
    - The output after applying scaled dot product attention.
    - The attention weights.
    """
    depth = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(depth)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads."

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(p=0.1)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, value, key, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        values = self.values(value).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(key).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)

        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        device = queries.device
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale.to(device)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, values).transpose(1, 2).contiguous().view(N, -1, self.embed_size)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads,forward_expansion=2, dropout_rate=0.2):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

    def forward(self, value, key, query):
        norm_query = self.norm1(query)
        #attention = self.attention(value, key, query)
        attention = self.attention(value, key, norm_query)
        #x = query + attention
        #x = self.dropout(self.norm1(x))
        x = query + self.dropout(attention)

        norm_x = self.norm2(x)
        #forward = self.feed_forward(x)
        forward = self.feed_forward(norm_x)
        #out = x + forward
        #out = self.dropout(self.norm2(out))
        out = x + self.dropout(forward)
        return out

class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding .

    Args:
    - d_model (int): The dimension of the embeddings (and encoded positions).
    - max_len (int): The maximum length of the input sequences.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encodings to input embeddings.

        Args:
        - x (torch.Tensor): The input embeddings of shape (Batch size, Sequence length, d_model).

        Returns:
        - torch.Tensor: The input embeddings with added positional encodings.
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class LearnablePositionalEncoding(nn.Module):
    """
    Implements learnable positional encoding.

    Args:
    - d_model (int): The dimension of the embeddings (and encoded positions).
    - max_len (int): The maximum length of the input sequences.
    """
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        """
        Adds learnable positional encodings to input embeddings.

        Args:
        - x (torch.Tensor): The input embeddings of shape (Batch size, Sequence length, d_model).

        Returns:
        - torch.Tensor: The input embeddings with added learnable positional encodings.
        """
        return x + self.position_embeddings[:, :x.size(1)]

class AudioClassifierWithTransformer(nn.Module):
    def __init__(self, num_classes, input_length, embed_size, num_heads, num_encoder_layers=3):
        super(AudioClassifierWithTransformer, self).__init__()
        self.embed_size = embed_size
        self.conv_base = AudioFeatureExtractor(input_channels=1, sequence_length=input_length)
        #self.positional_encoding = PositionalEncoding(d_model=embed_size)
        self.positional_encoding = LearnablePositionalEncoding(d_model=embed_size)

        self.seq_length = self.determine_seq_length()

        self.transformer_encoders = nn.ModuleList(
            [TransformerBlock(embed_size, num_heads) for _ in range(num_encoder_layers)]
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.fc_out = nn.Linear(embed_size, num_classes)

    def determine_seq_length(self):
        return self.conv_base.calculate_feature_dim(16000) // self.embed_size

    def forward(self, x):
        conv_features = self.conv_base(x)
        batch_size = conv_features.size(0)

        conv_features = conv_features.view(batch_size, self.seq_length, self.embed_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, conv_features), dim=1)
        x = self.positional_encoding(x)

        for encoder in self.transformer_encoders:
            x = encoder(x, x, x)

        cls_token_final = x[:, 0]
        out = self.fc_out(cls_token_final)

        return out

