from torch.utils.data import Dataset,WeightedRandomSampler
import tqdm
import torch
import random
import pdb
import numpy as np

class ELECTRADataset(Dataset):
    def __init__(self, samples_path, embedding_path,labels_path):
        self.embeddings = np.load(embedding_path)
        self.samples = np.load(samples_path)
        self.labels = np.load(labels_path)
        self.seq_len = self.samples.shape[1]+1
        #Initialize cls token vector values

        #take average of all embeddings
        self.cls = np.average(self.embeddings,axis=0)

        self.frequency_index = self.samples.shape[2] - 1 
        self.cls_frequency = 1



        #initialize mask token vector values

        #find max and min ranges of values for every feature in embedding space
        #create random embedding
        self.embedding_mins = np.amin(self.embeddings,axis=0)
        self.embedding_maxes = np.amin(self.embeddings,axis=0)
        self.mask = self.generate_random_embedding()


        self.padding = np.zeros(self.embeddings.shape[1])
        #add cls, mask, and padding embeddings to vocab embeddings
        self.embeddings = np.concatenate((self.embeddings,np.expand_dims(self.mask,axis=0)))
        self.embeddings = np.concatenate((self.embeddings,np.expand_dims(self.cls,axis=0)))
        self.embeddings = np.concatenate((self.embeddings,np.expand_dims(self.padding,axis=0)))
        
        self.mask_index = self.lookup_embedding(self.mask)
        self.cls_index = self.lookup_embedding(self.cls)
        self.padding_index = self.lookup_embedding(self.padding)
        

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        
        #pdb.set_trace()
        sample = self.samples[item]
        sorted_indices = np.argsort(sample[:,1])
        sample = sample[sorted_indices][::-1]
        cls_marker = np.array([[self.cls_index,self.cls_frequency]],dtype=np.float)
        sample = np.concatenate((cls_marker,sample))
        electra_input,frequencies = self.match_sample_to_embedding(sample)
        electra_label = self.labels[item]

        output = {"electra_input": torch.tensor(electra_input,dtype=torch.long),
                "electra_label": torch.tensor(electra_label,dtype=torch.long),
                "species_frequencies": torch.tensor(frequencies,dtype=torch.long),
                }

        return output

    def match_sample_to_embedding(self, sample):
        electra_input = sample[:,0].copy()
        frequencies = np.zeros(sample.shape[0])
        for i in range(sample.shape[0]):
            #pdb.set_trace()
            if sample[i,self.frequency_index] > 0:
                frequencies[i] = sample[i,self.frequency_index]
            else:
                electra_input[i] = self.padding_index
                

        return electra_input,frequencies

    def generate_random_frequency(self):
        return np.random.randint(self.frequency_min,self.frequency_max)

    def generate_random_embedding(self):
        return np.random.uniform(self.embedding_mins,self.embedding_maxes)

    def vocab_len(self):
        return self.embeddings.shape[0]

    def lookup_embedding(self,bug):
        return np.where(np.all(self.embeddings == bug,axis=1))[0][0]

def create_weighted_sampler(labels_path):
    labels = np.load(labels_path)
    labels_unique, counts = np.unique(labels,return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    example_weights = [class_weights[int(e)] for e in labels]
    sampler = WeightedRandomSampler(example_weights,len(labels))
    return sampler
