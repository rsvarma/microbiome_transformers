from transformers import ElectraConfig,ElectraForPreTraining,ElectraForMaskedLM
import torch.nn as nn
import torch
import pdb

class ElectraDiscriminator(nn.Module):

    def __init__(self,config:ElectraConfig,embeddings):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.embedding_size,padding_idx = config.vocab_size-1)
        self.embed_layer.weight = nn.Parameter(embeddings)
        self.discriminator = ElectraForPreTraining(config)
        self.sigmoid = nn.Sigmoid()

    def forward(self,data,attention_mask,labels):
        #pdb.set_trace()
        data = self.embed_layer(data)
        loss,scores = self.discriminator(attention_mask=attention_mask,inputs_embeds=data,labels=labels)
        scores = self.sigmoid(scores)
        return loss, scores


    


class ElectraGenerator(nn.Module):

    def __init__(self,config: ElectraConfig,embeddings,generator = None, embed_layer = None):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.embedding_size, padding_idx= config.vocab_size-1)
        if embed_layer:
            self.embed_layer.load_state_dict(torch.load(embed_layer))
        else:
            self.embed_layer.weight = nn.Parameter(embeddings)
        if generator:
            self.generator = ElectraForMaskedLM.from_pretrained(generator,config=config)  
        else:      
            self.generator = ElectraForMaskedLM(config)
        self.softmax = nn.Softmax(dim=2)

    def forward(self,data,attention_mask,labels):
        #pdb.set_trace()
        data = self.embed_layer(data)
        loss,scores = self.generator(attention_mask=attention_mask,inputs_embeds=data,labels=labels)
        scores = self.softmax(scores)
        return loss, scores