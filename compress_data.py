import numpy as np

import sys
import pdb

if len(sys.argv) == 4:
    data = np.load(sys.argv[1])
    sample_len = int(sys.argv[2])
    save_file = sys.argv[3]
else:
    print("usage: python compress_data.py <data> <cutoff> <save file>")
    quit()



sorted_data = np.zeros(data.shape)
for i in range(data.shape[0]):
    sorted_indices = np.argsort(data[i,:,1])
    sorted_sample = data[i,sorted_indices,:][::-1]
    sorted_data[i] = sorted_sample

compressed_data = sorted_data[:,:sample_len,:]

np.save(save_file,compressed_data)





