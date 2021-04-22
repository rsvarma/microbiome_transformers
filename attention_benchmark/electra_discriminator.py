from transformers import ElectraConfig,ElectraForSequenceClassification
import torch.nn as nn
import torch
import pdb
import torch.nn.functional as F
import math
def attention(q, k, v, d_k, mask=None, dropout=None):
    #pdb.set_trace()
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output


class UnchangedValueAttention(nn.Module):

    def __init__(self,input_dim):
        super().__init__()
        self.WQ = nn.Linear(input_dim,input_dim)
        self.WK = nn.Linear(input_dim,input_dim)
        self.input_dim = input_dim

    def forward(self,data,mask):
        #pdb.set_trace()
        q_s = self.WQ(data)
        k_s = self.WK(data)
        x = attention(q_s,k_s,data,self.input_dim,mask)
        return x

class ChangedValueAttention(nn.Module):

    def __init__(self,input_dim):
        super().__init__()
        self.WQ = nn.Linear(input_dim,input_dim)
        self.WK = nn.Linear(input_dim,input_dim)
        self.WV = nn.Linear(input_dim,input_dim)
        self.input_dim = input_dim

    def forward(self,data,mask):
        #pdb.set_trace()
        q_s = self.WQ(data)
        k_s = self.WK(data)
        v_s = self.WV(data)
        x = attention(q_s,k_s,v_s,self.input_dim,mask)
        return x       

class EncoderNoProject(nn.Module):

    def __init__(self,input_dim):
        super().__init__()
        self.WQ = nn.Linear(input_dim,input_dim)
        self.WK = nn.Linear(input_dim,input_dim)
        self.WV = nn.Linear(input_dim,input_dim)
        self.input_dim = input_dim
        self.dropout = nn.Dropout(p=0.1)
        self.outputdense = nn.Linear(input_dim,input_dim)
        self.norm = nn.LayerNorm((input_dim,), eps=1e-12, elementwise_affine=True)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self,data,mask):
        #pdb.set_trace()
        q_s = self.WQ(data)
        k_s = self.WK(data)
        v_s = self.WV(data)
        x = attention(q_s,k_s,v_s,self.input_dim,mask)
        x = self.dropout(x)
        x = self.outputdense(x)
        x = self.norm(x)
        x = self.dropout2(x)
        return x            

class EncoderProjectDouble(nn.Module):

    def __init__(self,input_dim):
        super().__init__()
        self.double_input_dim = input_dim*2
        self.project = nn.Linear(input_dim,self.double_input_dim)
        self.WQ = nn.Linear(self.double_input_dim,self.double_input_dim)
        self.WK = nn.Linear(self.double_input_dim,self.double_input_dim)
        self.WV = nn.Linear(self.double_input_dim,self.double_input_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.outputdense = nn.Linear(self.double_input_dim,self.double_input_dim)
        self.norm = nn.LayerNorm((self.double_input_dim,), eps=1e-12, elementwise_affine=True)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self,data,mask):
        #pdb.set_trace()
        data = self.project(data)
        q_s = self.WQ(data)
        k_s = self.WK(data)
        v_s = self.WV(data)
        x = attention(q_s,k_s,v_s,self.double_input_dim,mask)
        x = self.dropout(x)
        x = self.outputdense(x)
        x = self.norm(x)
        x = self.dropout2(x)
        return x                


class EncoderProjectDoubleIntermediate(nn.Module):

    def __init__(self,input_dim):
        super().__init__()
        self.double_input_dim = input_dim*2
        self.project = nn.Linear(input_dim,self.double_input_dim)
        self.WQ = nn.Linear(self.double_input_dim,self.double_input_dim)
        self.WK = nn.Linear(self.double_input_dim,self.double_input_dim)
        self.WV = nn.Linear(self.double_input_dim,self.double_input_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.outputdense = nn.Linear(self.double_input_dim,self.double_input_dim)
        self.norm = nn.LayerNorm((self.double_input_dim,), eps=1e-12, elementwise_affine=True)
        self.dropout2 = nn.Dropout(p=0.1)
        self.intermediate = nn.Linear(self.double_input_dim,self.double_input_dim*2)
        self.inverseintermediate = nn.Linear(self.double_input_dim*2,self.double_input_dim)
        self.norm2= nn.LayerNorm((self.double_input_dim,), eps=1e-12, elementwise_affine=True)
        self.dropout3 = nn.Dropout(p=0.1)


    def forward(self,data,mask):
        #pdb.set_trace()
        data = self.project(data)
        q_s = self.WQ(data)
        k_s = self.WK(data)
        v_s = self.WV(data)
        x = attention(q_s,k_s,v_s,self.double_input_dim,mask)
        x = self.dropout(x)
        x = self.outputdense(x)
        x = self.norm(x)
        x = self.dropout2(x)
        x = self.intermediate(x)
        x = self.inverseintermediate(x)
        x = self.norm2(x)
        x = self.dropout3(x)
        return x                    


class AttentionModel(nn.Module):

    def __init__(self, embeddings, embed_layer = None):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=embeddings.shape[0],embedding_dim=embeddings.shape[1],padding_idx = embeddings.shape[0]-1)
        if embed_layer:
            self.embed_layer.load_state_dict(torch.load(embed_layer))
        else:
            self.embed_layer.weight = nn.Parameter(embeddings)
        self.attention = EncoderProjectDoubleIntermediate(embeddings.shape[1])
        self.fc1 = nn.Linear(embeddings.shape[1]*2,1)


    def forward(self,data,attention_mask):
        #pdb.set_trace()
        data = self.embed_layer(data)
        embeds = self.attention(data,attention_mask)
        scores = self.fc1(embeds[:,0,:])
        scores = torch.sigmoid(scores)
        return scores


class ElectraModel(nn.Module):

    def __init__(self, embeddings, embed_layer = None):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=embeddings.shape[0],embedding_dim=embeddings.shape[1],padding_idx = embeddings.shape[0]-1)
        if embed_layer:
            self.embed_layer.load_state_dict(torch.load(embed_layer))
        else:
            self.embed_layer.weight = nn.Parameter(embeddings)
        self.attention = EncoderProjectDoubleIntermediate(embeddings.shape[1])
        self.fc1 = nn.Linear(embeddings.shape[1]*2,embeddings.shape[1]*2)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(embeddings.shape[1]*2,1)


    def forward(self,data,attention_mask):
        #pdb.set_trace()
        data = self.embed_layer(data)
        embeds = self.attention(data,attention_mask)
        scores = self.fc1(embeds[:,0,:])
        scores = self.dropout(scores)
        scores = self.fc2(scores)
        scores = torch.sigmoid(scores)
        return scores        


class ElectraModelCrossEntropy(nn.Module):

    def __init__(self, embeddings, embed_layer = None):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=embeddings.shape[0],embedding_dim=embeddings.shape[1],padding_idx = embeddings.shape[0]-1)
        if embed_layer:
            self.embed_layer.load_state_dict(torch.load(embed_layer))
        else:
            self.embed_layer.weight = nn.Parameter(embeddings)
        self.attention = EncoderProjectDoubleIntermediate(embeddings.shape[1])
        self.fc1 = nn.Linear(embeddings.shape[1]*2,embeddings.shape[1]*2)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(embeddings.shape[1]*2,2)
        self.softmax = nn.Softmax()



    def forward(self,data,attention_mask):
        #pdb.set_trace()
        data = self.embed_layer(data)
        embeds = self.attention(data,attention_mask)
        scores = self.fc1(embeds[:,0,:])
        scores = self.dropout(scores)
        scores = self.fc2(scores)
        scores = self.softmax(scores)
        return scores          