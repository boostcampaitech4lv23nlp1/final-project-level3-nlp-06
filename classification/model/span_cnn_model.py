import torch.nn as nn
from transformers import AutoModel


class SpanDetectionCNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(SpanDetectionCNN, self).__init__()
        
        self.config = config
        
        self.Embedding = nn.Embedding(vocab_size, 768)
        if self.config["get_bert_embedding"]:
            embedding_weights = self.prepare_embeddings()
            self.Embedding.weight = embedding_weights
        self.Layer = nn.Sequential(
            nn.Conv1d(768, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 2, kernel_size=3, padding=1),
            nn.Softmax(),
        )
        
    def prepare_embeddings(self):
        transformer = AutoModel.from_pretrained(self.config["model_name"])
        for name, param in transformer.named_parameters():
            if "embeddings.word_embeddings" in name:
                embedding_weights = param
        del transformer
        return embedding_weights
        
    def forward(self, inputs):
        x = self.Embedding(inputs)
        x = x.transpose(1, 2)
        return self.Layer(x).squeeze(1)
    