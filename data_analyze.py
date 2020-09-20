import numpy as np
import matplotlib.pyplot as plt 
import sys


if len(sys.argv) == 3:
    train_data = np.load(sys.argv[1])
    test_data = np.load(sys.argv[2])
else:
    print("usage: python data_analyze.py <train_data> <test_data>")
    quit()

all_data = np.concatenate((train_data,test_data))

sample_lengths = np.zeros(all_data.shape[0])
for i in range(all_data.shape[0]):
    length = 0
    for j in range(all_data.shape[1]):
        if all_data[i,j,1] > 0:
            length += 1
    sample_lengths[i] = length

plt.hist(sample_lengths,range(0,2000))
plt.show()



