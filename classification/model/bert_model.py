from transformers import AutoModelForSequenceClassification


def multi_label_bert(config, vocab_size=None):
    return AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        problem_type="multi_label_classification"
    )
    
def bi_label_bert(config, vocab_size=None):
    return AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
    )