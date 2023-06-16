from torch import nn
from constants import *
from layers import *

import torch


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model):
        super().__init__()

        self.encoder = Encoder(src_vocab_size, d_model)
        self.decoder = Decoder(trg_vocab_size, d_model)

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        e_output = self.encoder(src_input, e_mask) # (B, L, d_model)
        output = self.decoder(trg_input, e_output, e_mask, d_mask) # B, L, trg_vocab_size)

        return output


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder()
        self.layers = nn.ModuleList([EncoderLayer() for i in range(encoder_num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_mask):
        x = self.src_embedding(x) # (B, L) => (B, L, d_model)
        x = self.positional_encoder(x) # (B, L, d_model) => (B, L, d_model)
        for i in range(encoder_num_layers):
            x = self.layers[i](x, e_mask) # (B, L, d_model)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, d_model):
        super().__init__()
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder()
        self.layers = nn.ModuleList([DecoderLayer() for i in range(decoder_num_layers)])
        self.layer_norm = LayerNormalization()
        self.output_linear = nn.Linear(d_model, trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, e_output, e_mask, d_mask):
        x = self.trg_embedding(x) # (B, L) => (B, L, d_model)
        x = self.positional_encoder(x) # (B, L, d_model) => (B, L, d_model)
        for i in range(decoder_num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask) # (B, L, d_model)

        x = self.layer_norm(x)
        output = self.softmax(self.output_linear(x)) # (B, L, d_model) => # (B, L, trg_vocab_size)

        return output