from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModel
import torch.nn as nn

    
def transformer(config, vocab_size=None):
    return AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=1,
    )
    
def span_transformer(config, vocab_size=None):
    return AutoModelForTokenClassification.from_pretrained(
        config["model_name"],
        num_labels=2
    )
    
class Token_Sequence_transformer(nn.Module):
    def __init__(self, config, vocab_size=None):
        super(Token_Sequence_transformer, self).__init__()
        self.transformer = AutoModel.from_pretrained(config["model_name"])
        self.sequence_classification = nn.Linear(256, 1)
        self.token_classification = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden_states = self.transformer(input_ids, attention_mask, token_type_ids).last_hidden_state
        
        token_output = self.token_classification(hidden_states)
        
        cls_output = hidden_states[:, 0, :]
        sequence_output = self.sequence_classification(cls_output)
        sequence_output = self.sigmoid(sequence_output)
        
        return token_output, sequence_output
    