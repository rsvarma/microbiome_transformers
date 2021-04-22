from transformers import ElectraConfig,ElectraModel
import torch.nn as nn
import torch
import pdb

class ElectraDiscriminator(nn.Module):

    def __init__(self,config:ElectraConfig,num_tasks,embeddings,discriminator = None, embed_layer = None):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.embedding_size,padding_idx = config.vocab_size-1)
        if embed_layer:
            self.embed_layer.load_state_dict(torch.load(embed_layer))
        else:
            self.embed_layer.weight = nn.Parameter(embeddings)
        if discriminator:
            self.discriminator = ElectraModel.from_pretrained(discriminator,config=config)
        else:
            self.discriminator = ElectraModel(config)
        self.classificationhead = nn.Sequential(
                                    nn.Linear(config.hidden_size,config.hidden_size),
                                    nn.LayerNorm(config.hidden_size),
                                    nn.Linear(config.hidden_size,2)
                                )
        self.heads = nn.ModuleList([self.classificationhead for i in range(num_tasks)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self,data,attention_mask):
        #pdb.set_trace()
        data = self.embed_layer(data)
        output = self.discriminator(attention_mask=attention_mask,inputs_embeds=data)[0]
        scores = []
        for head in self.heads:
            score = head(output[:,0,:])
            scores.append(self.softmax(score))
        return scores