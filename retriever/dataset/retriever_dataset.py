from transformers import AutoTokenizer
import random
import json


class KOLD_dataset:
    def __init__(self, config, data_path):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        with open(data_path, "r", encoding='utf-8') as f:
            self.data = json.load(f) # [{"title": title, "positive": [p1(str), ...], "negative": [n1(str), ...]}]
        self.negatives = self.organize_with_negative(self.data) # {"target": [n1(str), n2(str)...], "id": [0, 0, ...]}
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]
        self.position = 0
            
    def organize_with_negative(self, datas):
        '''
        Get negative sentence and id from self.data for construct in-batch-negative.
        Object for this function is preventing take negative pairs from same title.
        "target": Negative sentences.
        "id": Index of where negative sentence came from.
        '''
        negatives = {"target": [], "id": []}
        for i, data in enumerate(datas):
            negatives["target"] += data["negative"]
            negatives["id"] += [i for _ in range(len(data["negative"]))]
        return negatives

    def __len__(self):
        return len(self.negatives['target'])//self.batch_size
        
    def tokenize(self, sentence):
        tokenized_sentence = self.tokenizer(
            sentence,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        return tokenized_sentence
    
    def __iter__(self):
        if self.shuffle:
            shuffled_index = list(range(len(self)))
            random.shuffle(shuffled_index)
            self.negatives = {
                "target": [self.negatives["target"][i] for i in shuffled_index], 
                "id": [self.negatives["id"][i] for i in shuffled_index]
                }
        for idx in range(len(self)//self.batch_size):
            negatives = self.negatives["target"][idx*self.batch_size:(idx+1)*self.batch_size]
            ids = self.negatives["id"][idx*self.batch_size:(idx+1)*self.batch_size]
            positives = [random.choice(self.data[id]['positive']) for id in ids]
            negatives = self.tokenize(negatives)
            positives = self.tokenize(positives)
            yield negatives, positives
