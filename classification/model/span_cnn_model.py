import torch
import torch.nn as nn
from transformers import AutoModel


class SpanDetectionCNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(SpanDetectionCNN, self).__init__()
        
        self.config = config
        
        self.Embedding = nn.Embedding(vocab_size, 768)

        self.Layer = nn.Sequential(
            nn.Conv1d(768, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 2, kernel_size=3, padding=1),
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def forward(self, inputs, mask):
        x = self.Embedding(inputs)
        x = x.transpose(1, 2)
        y = self.Layer(x)
        pooled_outputs = self.mean_pooling(y[:, 1, :], mask)
        pooled_outputs = self.sigmoid(pooled_outputs)
        
        return y, pooled_outputs
    