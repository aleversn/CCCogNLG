import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration

class ProphetNet(nn.Module):

    def __init__(self, tokenizer, pretrained_dir):
        super().__init__()
        self.model = ProphetNetForConditionalGeneration.from_pretrained(pretrained_dir)
        self.tokenizer = tokenizer
    
    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)

        logits = outputs.logits
        lprobs = F.log_softmax(
            logits.view(-1, logits.size(-1)),
            dim=-1,
            dtype=torch.float32,
        )

        loss = F.nll_loss(
            lprobs,
            decoder_input_ids.view(-1),
            reduction='sum',
            ignore_index=102,
        )

        return (loss, logits)