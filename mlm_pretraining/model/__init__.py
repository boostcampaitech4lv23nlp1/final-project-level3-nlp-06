from transformers import ElectraForMaskedLM, ElectraForPreTraining
import torch.nn.functional as F
import torch.nn as nn
import torch


class ELECTRA(nn.Module):
    def __init__(self, generator_name, discriminator_name, mask_token_id):
        super(ELECTRA, self).__init__()
        self.generator = ElectraForMaskedLM.from_pretrained(generator_name)
        self.discriminator = ElectraForPreTraining.from_pretrained(discriminator_name)
        self.mask_token_id = mask_token_id
        
    def forward(self, inputs):
        outputs = self.generator(**inputs).logits
        
        # Find Index of Masked Token.
        batch, idx = (inputs['input_ids'] == self.mask_token_id).nonzero(as_tuple=True)
        generated_outputs = inputs['input_ids']
        for b, i in zip(batch, idx):
            generated_outputs[b][i] = torch.argmax(outputs[b][i], dim=0)
        
        outputs = self.discriminator(generated_outputs).logits
        outputs = F.sigmoid(outputs)
        
        return outputs
    