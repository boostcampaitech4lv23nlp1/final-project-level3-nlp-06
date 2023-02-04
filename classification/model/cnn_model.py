import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, vocab_size):
        super(CNNModel, self).__init__()
        
        self.Embedding = nn.Embedding(vocab_size, 128)
        self.res_block1 = self.ResBlock(128, 128)
        self.up_conv1 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.res_block2 = self.ResBlock(256, 256)
        self.up_conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.res_block3 = self.ResBlock(512, 512)

        self.linear = nn.Conv1d(512, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def ResBlock(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
        )
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def forward(self, inputs, mask):
        x = self.Embedding(inputs)
        x = x.transpose(1, 2)
        residual = x
        x = self.res_block1(x)
        x += residual
        x = self.up_conv1(x)
        
        residual = x
        x = self.res_block2(x)
        x += residual
        x = self.up_conv2(x)
        
        residual = x
        x = self.res_block3(x)
        x += residual
        
        x = self.linear(x)
        pooled_outputs = self.mean_pooling(x.squeeze(1), mask)
        outputs = self.sigmoid(pooled_outputs)
        return outputs
        