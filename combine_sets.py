import sys
import argparse
import pdb
import numpy as np
from progressbar import ProgressBar

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--otu_files", type=str,nargs='+', default=None, help="paths to otus for different tasks")
    parser.add_argument("--labels", type=str,nargs='+', default=None, help="paths to labels for different tasks, in same order as for --otu_files")
    parser.add_argument("--otu_output",type=str,default=None,help="output file to save combined otu in")
    parser.add_argument("--label_output",type=str,default=None,help="output file to save combined labels in")
    args = parser.parse_args()

    combined_otu = []
    combined_labels = []
    total_files = len(args.otu_files)
    first_task = np.load(args.otu_files[0])
    first_label = np.load(args.labels[0])
    for x,label in zip(first_task,first_label):
        combined_otu.append(x)
        new_label = np.full(total_files,-100,dtype="int64")
        new_label[0] = label.item()
        combined_labels.append(new_label)


    for i in range(1,len(args.otu_files)):
        comp_task = np.load(args.otu_files[i])
        comp_label = np.load(args.labels[i])
        pbar = ProgressBar()
        for x in pbar(range(comp_task.shape[0])):
            sample = comp_task[x]
            sample_label = comp_label[x]
            match = False
            for y in range(len(combined_otu)):
                if np.array_equal(sample,combined_otu[y]):
                    combined_labels[y][i] = sample_label.item()
                    match = True
            if not match:
                combined_otu.append(sample)
                new_label = np.full(total_files,-100,dtype="int64")
                new_label[i] = sample_label
                combined_labels.append(new_label)

    final_otu = np.stack(combined_otu)
    final_labels = np.stack(combined_labels)
    pdb.set_trace()
    np.save(args.otu_output,final_otu)
    np.save(args.label_output,final_labels)
    print("completed")



if __name__ == "__main__":
    main()