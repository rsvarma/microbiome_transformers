from torch.utils.data import Dataset
import tqdm
import torch
import random
import pdb
import numpy as np

class ELECTRADataset(Dataset):
    def __init__(self, samples_path, embedding_path,input_embed=True):
        self.embeddings = np.load(embedding_path)
        self.samples = np.load(samples_path)
        self.seq_len = self.samples.shape[1]+1
        self.input_embed = input_embed
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
        electra_input,electra_label,frequencies,mask_locations = self.match_sample_to_embedding(sample)

        if self.input_embed:
            output = {"electra_input": torch.tensor(electra_input,dtype=torch.float),
                    "electra_label": torch.tensor(electra_label,dtype=torch.long),
                    "species_frequencies": torch.tensor(frequencies,dtype=torch.long),
                    "mask_locations": torch.tensor(mask_locations)
                    }
        else:
            output = {"electra_input": torch.tensor(electra_input,dtype=torch.long),
                    "electra_label": torch.tensor(electra_label,dtype=torch.long),
                    "species_frequencies": torch.tensor(frequencies,dtype=torch.long),
                    "mask_locations": torch.tensor(mask_locations)
                    }

        return output

    def match_sample_to_embedding(self, sample):
        output_label = []
        if self.input_embed:
            electra_input = np.zeros((sample.shape[0],self.embeddings.shape[1]))
        else:
            electra_input = sample[:,0]
        frequencies = np.zeros(sample.shape[0])
        mask_locations = np.full(sample.shape[0],False)
        masked = False
        for i in range(sample.shape[0]):
            #pdb.set_trace()
            if sample[i,self.frequency_index] > 0:
                if self.input_embed:
                    electra_input[i] = self.embeddings[int(sample[i,0])]
                frequencies[i] = sample[i,self.frequency_index]
                prob = random.random()
                if prob < 0.15 and i > 0 and sample[i,self.frequency_index] > 100:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8  and masked == False:
                        if self.input_embed:
                            electra_input[i] = self.mask
                        else:
                            electra_input[i] = self.mask_index
                        mask_locations[i] = True
                        output_label.append(1)
                        #electra, so not limiting masks
                        #masked = True


                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        if self.input_embed:
                            electra_input[i] = self.embeddings[random.randrange(self.embeddings.shape[0])]
                        else:
                            electra_input[i] = random.randrange(self.embeddings.shape[0])
                        mask_locations[i] = True
                        output_label.append(1)
                        #append index of embedding to output label
                        
                    
                    # 10% randomly change token to current token
                    else:
                        mask_locations[i] = False
                        output_label.append(0)


                else:
                    output_label.append(0)

            else:
                output_label.append(0)

        return electra_input, output_label,frequencies,mask_locations

    def generate_random_frequency(self):
        return np.random.randint(self.frequency_min,self.frequency_max)

    def generate_random_embedding(self):
        return np.random.uniform(self.embedding_mins,self.embedding_maxes)

    def vocab_len(self):
        return self.embeddings.shape[0]

    def lookup_embedding(self,bug):
        return np.where(np.all(self.embeddings == bug,axis=1))[0][0]