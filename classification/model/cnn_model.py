import torch
import torch.nn as nn
from transformers import AutoModel


class CNNModel(nn.Module):
    def __init__(self, config, vocab_size):
        super(CNNModel, self).__init__()
        self.config = config
        
        self.Embedding = nn.Embedding(vocab_size, 768)
        self.Layer = nn.Sequential(
            self.conv_block(128, 64), #384
            self.conv_block(64, 32), # 192
            self.conv_block(32, 1), # 96
            nn.Linear(96, 1),
            nn.Sigmoid()
        )
        
    def conv_block(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
    def forward(self, inputs, mask):
        x = self.Embedding(inputs)
        return self.Layer(x)
    

class VerifiableCNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(VerifiableCNN, self).__init__()
        
        self.config = config
        
        self.Embedding = nn.Embedding(vocab_size, 768)
        self.Layer = nn.Sequential(
            nn.Conv1d(768, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.sigmoid = nn.Sigmoid()
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def forward(self, inputs, mask):
        x = self.Embedding(inputs)
        x = x.transpose(1, 2)
        x = self.Layer(x)
        pooled_outputs = self.mean_pooling(x.squeeze(1), mask)
        pooled_outputs = self.sigmoid(pooled_outputs)
        return pooled_outputs
        