from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch.nn as nn
import torch


class Token_Sequence_transformer(nn.Module):
    def __init__(self, config, vocab_size=None):
        super(Token_Sequence_transformer, self).__init__()
        self.transformer = AutoModel.from_pretrained(config["model_name"])
        self.sequence_classification = nn.Linear(256, 9)
        self.token_classification = nn.Linear(256, 2)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden_states = self.transformer(input_ids, attention_mask, token_type_ids).last_hidden_state
        
        token_output = self.token_classification(hidden_states)
        
        cls_output = hidden_states[:, 0, :]
        sequence_output = self.sequence_classification(cls_output)
        
        return token_output, sequence_output
    

class ClassificationModel:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.SequenceClassifier = AutoModelForSequenceClassification.from_pretrained(
            config["sequence_model"]
        )
        self.SequenceClassifier.eval()
        self.SequenceClassifier.to(self.device)

        self.TokenClassifier = Token_Sequence_transformer(config)
        self.TokenClassifier.load_state_dict(
            torch.load(config["token_model"])
        )
        self.TokenClassifier.eval()
        self.TokenClassifier.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        
    def tokenize(self, sentence):
        # 배치 단위로 수행할 것이라면 padding 등의 옵션을 추가해 줘야할 것.
        return self.tokenizer(
            sentence,
            return_tensors="pt"
        )
        
    def predict_token(self, tokenized_sentence):
        with torch.no_grad():
            token_output, _ = self.TokenClassifier(**tokenized_sentence)
        token_output = torch.softmax(token_output, dim=-1)[:, :, 1]
        return token_output
        
    def predict(self, sentence):
        data = {k: v.to(self.device) for k, v in self.tokenize(sentence).items()}
        hate = False
        with torch.no_grad():
            seq_output = self.SequenceClassifier(**data).logits.cpu().squeeze()
            seq_output = seq_output.tolist()
        if self.config["hate_threshold"] < seq_output:
            hate = True
            token_output = self.predict_token(data).cpu().squeeze()
            token_output = token_output.tolist()[1:-1]
            tokenized_sentence = self.tokenizer.tokenize(sentence)
        else:
            token_output = None
            tokenized_sentence = None
            
        return hate, token_output, tokenized_sentence