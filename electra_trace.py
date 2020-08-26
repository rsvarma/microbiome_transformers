from transformers import ElectraTokenizer, ElectraForPreTraining
import torch
import pdb

tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator')
pdb.set_trace()
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
scores = model(input_ids)[0]