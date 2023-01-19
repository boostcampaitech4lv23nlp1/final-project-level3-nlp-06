from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

    
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
    