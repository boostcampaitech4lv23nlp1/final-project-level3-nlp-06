from transformers import AutoModelForSequenceClassification

    
def transformer(config, vocab_size=None):
    return AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=1,
    )