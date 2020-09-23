from transformers import ElectraConfig,ElectraForPreTraining,ElectraForMaskedLM
import torch.nn as nn
import torch
import pdb

class ElectraPretrainModel(nn.Module):

    def __init__(self,config: ElectraConfig,embeddings):
        #pdb.set_trace()
        super().__init__()
        self.generator = ElectraGenerator(config,embeddings)
        self.discriminator = ElectraDiscriminator(config,embeddings)

    def forward(self,data,attention_mask):
            #pdb.set_trace()
            g_loss, g_scores = self.generator(data['electra_input'],attention_mask,data['electra_mask_label'])
            #pdb.set_trace()
            with torch.no_grad():
                predictions = g_scores.max(2)[1]
                disc_labels = (data["electra_label"] != predictions).long()
                disc_labels = disc_labels.masked_fill(~data["mask_locations"],0)
                disc_inputs = torch.where(data["mask_locations"],predictions,data["electra_input"])
            

            d_loss,d_scores = self.discriminator(disc_inputs,attention_mask,disc_labels)
            electra_loss = g_loss+50.0*d_loss
            return electra_loss,d_scores,g_scores,disc_labels


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

    def __init__(self,config: ElectraConfig,embeddings):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.embedding_size, padding_idx= config.vocab_size-1)
        self.embed_layer.weight = nn.Parameter(embeddings)        
        self.generator = ElectraForMaskedLM(config)
        self.softmax = nn.Softmax(dim=2)

    def forward(self,data,attention_mask,labels):
        #pdb.set_trace()
        data = self.embed_layer(data)
        loss,scores = self.generator(attention_mask=attention_mask,inputs_embeds=data,labels=labels)
        scores = self.softmax(scores)
        return loss, scores