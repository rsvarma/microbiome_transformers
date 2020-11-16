import numpy as np
import matplotlib.pyplot as plt 
import sys
import pdb

if len(sys.argv) == 4:
    train_data = np.load(sys.argv[1])
    test_data = np.load(sys.argv[2])
    sample_len = int(sys.argv[3])
else:
    print("usage: python compress_data.py <train_data> <test_data> <cutoff>")
    quit()



new_train = np.zeros(train_data.shape)
new_test = np.zeros(test_data.shape)
pdb.set_trace()
for i in range(train_data.shape[0]):
    sorted_indices = np.argsort(train_data[i,:,1])
    sorted_sample = train_data[i,sorted_indices,:][::-1]
    new_train[i] = sorted_sample
    
for i in range(test_data.shape[0]):
    sorted_indices = np.argsort(test_data[i,:,1])
    sorted_sample = test_data[i,sorted_indices,:][::-1]
    new_test[i] = sorted_sample
pdb.set_trace()
compressed_train = new_train[:,:sample_len,:]
compressed_test = new_test[:,:sample_len,:]

np.save('IBD_train_{}'.format(sample_len),compressed_train)
np.save('IBD_test_{}'.format(sample_len),compressed_test)




