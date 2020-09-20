##############################################################################
#Author: Rohan Varma
#
#Command Line tool for benchmarking results on AGP microbiome data.
#Allows testing of various sampleaggregation strategies including multiplying 
#counts by an embedding matrix, averaging together embeddings, and
#weighted averaging of embeddings. Also allows to use a single dense layer
#or a random forest classifier.
###############################################################################

import argparse
import sys
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
import sklearn
import numpy as np
import pdb
import tqdm
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
from inspect import signature

def create_weighted_sampler(labels):
    labels_unique, counts = np.unique(labels,return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    example_weights = [class_weights[int(e)] for e in labels]
    sampler = data_utils.WeightedRandomSampler(example_weights,len(labels))
    return sampler


def calc_auc(y_true,y_probas,show_plot=False):
    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_probas,pos_label = 1)
    auc_score = metrics.auc(fpr,tpr)
    if show_plot:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.show()    
    return auc_score




class Net(nn.Module):

    def __init__(self,input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim,1)

    def forward(self,x):
        x = torch.sigmoid(self.fc1(x))
        return x
        

def dense_layer(train_dataloader,test_dataloader,train_labels,test_labels,embed_dim,lr,log_file,epoch_num):
    with open(log_file,"w+") as f:
        f.write("EPOCH,MODE, AVG LOSS, TOTAL CORRECT, TOTAL ELEMENTS, ACCURACY, AUC, TOTAL POSITIVE CORRECT, TOTAL POSITIVE, ACCURACY\n")


    net = Net(embed_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),lr=lr)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    for epoch in range(epoch_num):
        total_train_correct = 0
        total_train_samples = 0 
        total_train_positive_correct = 0
        total_train_positive = 0
        total_test_correct = 0
        total_test_positive_correct = 0
        total_test_positive = 0
        train_scores = []
        test_scores = []
        cumulative_loss = 0

        #device = "cpu"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(train_dataloader,
                              desc="EP_%s:%d" % ("train", epoch),
                              total=len(train_dataloader),
                              bar_format="{l_bar}{r_bar}")        
        #pdb.set_trace()
        net.train()                              
        for inputs,labels in data_iter:
            inputs,labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(inputs.float()).flatten()
            loss = criterion(output,labels.float())
            loss.backward()
            optimizer.step()
            predictions = output >= 0.5
            train_scores.append(output.detach().cpu())

            cumulative_loss += loss.item()

            total_train_correct += torch.sum(predictions == labels).item()
            total_train_samples += labels.shape[0]

            positive_inds = labels.nonzero(as_tuple=True)
            total_train_positive_correct += torch.sum(predictions[positive_inds] == labels[positive_inds]).item()
            total_train_positive += labels.nonzero().shape[0]

        auc_score = calc_auc(train_labels,torch.cat(train_scores).numpy())
        with open(log_file,"a") as f:
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(epoch,"train",cumulative_loss/len(train_dataloader),total_train_correct,total_train_samples,total_train_correct/total_train_samples*100,auc_score,total_train_positive_correct,total_train_positive,total_train_positive_correct/total_train_positive*100))

        cumulative_loss = 0

        #pdb.set_trace()
        data_iter = tqdm.tqdm(test_dataloader,
                              desc="EP_%s:%d" % ("test", epoch),
                              total=len(test_dataloader),
                              bar_format="{l_bar}{r_bar}")       
        net.eval()   
        for inputs,labels in data_iter:
            inputs,labels = inputs.to(device), labels.to(device)
            output = net(inputs.float()).flatten()
            loss = criterion(output,labels.float())
            cumulative_loss += loss.item()
            predictions = output >= 0.5
            test_scores.append(output.detach().cpu())

            total_test_correct += torch.sum(predictions == labels).item()

            positive_inds = labels.nonzero(as_tuple=True)
            total_test_positive_correct += torch.sum(predictions[positive_inds] == labels[positive_inds]).item()
            total_test_positive += labels.nonzero().shape[0]
        auc_score = calc_auc(test_labels,torch.cat(test_scores).numpy())        
        with open(log_file,"a") as f:
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(epoch,"test",cumulative_loss/len(test_dataloader),total_test_correct,len(test_dataloader),total_test_correct/len(test_dataloader)*100,auc_score,total_test_positive_correct,total_test_positive,total_test_positive_correct/total_test_positive*100))


def computeMLstats(m, data, y, plot = False, plot_pr = False, graph_title = None, flipped = False):
    probs = m.predict_proba(data)
    
    #Flip for opposite class imbalance
    if flipped:
        y = [1 - i for i in y]
        probs = 1 - probs
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y, probs[:, 1])
    roc_auc = metrics.auc(fpr, tpr)

    
    #Compute precision-recall
    precision, recall, _ = metrics.precision_recall_curve(y, probs[:,1])

    #avg_pr = average_precision_score(precision, recall)
    average_precision = metrics.average_precision_score(y, probs[:,1])
    
    f1 = metrics.f1_score(y, np.argmax(probs, axis = 1))
    f2 = metrics.fbeta_score(y, np.argmax(probs, axis = 1), beta = 2)
    
    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='AUC ROC = %0.2f' %  roc_auc)
        #'AUC PR = %0.2f' % pr_avg_pr
        
        plt.legend(loc="lower right")
        x = np.linspace(0, 1, 10)
        plt.plot(x, x)
        plt.title(graph_title)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        
    if plot_pr:
        plt.subplot(1,2,2)
        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post', label='AUC PR = %0.2f' %  average_precision)
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.legend(loc="lower right")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
    
    return(roc_auc, fpr, tpr, average_precision, f1, f2)



def predictIBD(X_train, y_train, X_test, y_test, graph_title = "IBD Tests", max_depth = 12, n_estimators = 140, plot = True, plot_pr = True, weight = 20, feat_imp = False, flipped = False):
    weights = {0:1, 1:weight}
    m = RandomForestClassifier(max_depth= max_depth, random_state=0, n_estimators= n_estimators, class_weight = weights)
    m.fit(X_train, y_train)
    probs = m.predict_proba(X_test)
    probs_train = m.predict_proba(X_train)
    roc_auc, fpr, tpr, precision, f1, f2 = computeMLstats(m, data = X_test, y = y_test, plot = plot, plot_pr = plot_pr, graph_title = "IBD test", flipped = flipped)
    roc_auc, fpr, tpr, precision, f1, f2 = computeMLstats(m, data = X_train, y = y_train, plot = plot, plot_pr = plot_pr, graph_title = "IBD train", flipped = flipped)
    plt.show()

    #feat_imp_sort = getFeatureImportance(m, data = X_train, y = y_train)
    
    return(m, roc_auc, None, fpr, tpr, precision, f1, f2)#, feat_imp_sort)



def embed_matrix(train_freqs,test_freqs,embeds):
    train_freqs = np.arcsinh(train_freqs)
    test_freqs = np.arcsinh(test_freqs)
    train_samples = (np.matmul(train_freqs,embeds))
    test_samples = (np.matmul(test_freqs,embeds))    
    #train_samples = preprocessing.scale(np.matmul(train_freqs,embeds))
    #test_samples = preprocessing.scale(np.matmul(test_freqs,embeds))

    return train_samples,test_samples


def average(train_embeddings,test_embeddings):
    train_samples = preprocessing.scale(np.average(train_embeddings,axis=1))
    test_samples = preprocessing.scale(np.average(test_embeddings,axis=1))
    return train_samples, test_samples

def weighted_average(train_embeddings,test_embeddings,train_freqs,test_freqs):
    train_weights = train_freqs/train_freqs.sum(1)[:,None]
    test_weights = test_freqs/test_freqs.sum(1)[:,None]
    train_samples = np.zeros((train_embeddings.shape[0],train_embeddings.shape[2]))
    for i in range(train_embeddings.shape[0]):
        train_samples[i] = np.average(train_embeddings[i],axis=0,weights=train_weights[i])
    test_samples = np.zeros((test_embeddings.shape[0],test_embeddings.shape[2]))
    for i in range(test_embeddings.shape[0]):
        test_samples[i] = np.average(test_embeddings[i],axis=0,weights=test_weights[i])      
    return train_samples, test_samples

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-tr", "--train_dataset", required=True, type=str, help="path to numpy array for train asv counts")
    parser.add_argument("-te", "--test_dataset", required=True,type=str, help="path to numpy array for test asv counts")
    parser.add_argument("-ep", "--embedding_path", required=True, type=str, help="path to numpy array for GloVE embeddings")
    parser.add_argument("-trl","--train_label", required=True,type=str,help="path to numpy array for training set labels")
    parser.add_argument("-tel","--test_label",required=True,type=str,help="path to numpy array for testing set labels")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="path to output log file")
    parser.add_argument("-b", "--batch_size", type=int, default=48, help="batch size for training dense layer")
    parser.add_argument("-e", "--epoch_num", type = int, default=20, help="number of epochs to train dense layer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of dense layer")

    class_head = parser.add_mutually_exclusive_group()
    class_head.add_argument("--dense",action='store_true',help="choose dense layer as classification head")
    class_head.add_argument("--random_forest", action='store_true', help="choose random forest as classification head")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--average",action='store_true',help="select averaging as sample aggregation strategy")
    group.add_argument("--waverage",action='store_true',help="select weighted averaging as sample aggregation strategy")
    group.add_argument("--ematrix",action='store_true',help="select multiplication by embedded matrix as sample aggregation strategy")
    group.add_argument("--direct",action='store_true',help="select direct input as aggregation strategy")


    args = parser.parse_args()

    if not args.average and not args.waverage and not args.ematrix and not args.direct:
        parser.error('No aggregation strategy specified, add --average, --waverage, or --ematrix')


    if not args.dense and not args.random_forest:
        parser.error('No classification head type specified, add --dense or --random_forest')


    train_data = np.load(args.train_dataset)
    test_data = np.load(args.test_dataset)
    train_labels = np.load(args.train_label).flatten()
    test_labels = np.load(args.test_label).flatten()
    embeds = np.load(args.embedding_path)
    embed_dim = embeds.shape[1]
    log_file = args.output_path

    if not args.direct:
        train_embeddings = np.zeros((train_data.shape[0],embeds.shape[0],embed_dim))
        train_freqs = np.zeros((train_data.shape[0],embeds.shape[0]))
        for i in range(train_data.shape[0]):
            for j in range(train_data.shape[1]):
                if train_data[i,j,1] > 0:
                    train_embeddings[i,int(train_data[i,j,0])] = embeds[int(train_data[i,j,0])]
                    train_freqs[i,int(train_data[i,j,0])] = train_data[i,j,1]


        test_embeddings = np.zeros((test_data.shape[0],embeds.shape[0],embed_dim))
        test_freqs = np.zeros((test_data.shape[0],embeds.shape[0]))
        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[1]):
                if test_data[i,j,1] > 0:
                    test_embeddings[i,int(test_data[i,j,0])] = embeds[int(test_data[i,j,0])]
                    test_freqs[i,int(test_data[i,j,0])] = test_data[i,j,1]


    if args.average:
        train_samples,test_samples = average(train_embeddings,test_embeddings)
    elif args.waverage:
        train_samples,test_samples = weighted_average(train_embeddings,test_embeddings,train_freqs,test_freqs)
    elif args.ematrix:
        train_samples,test_samples = embed_matrix(train_freqs,test_freqs,embeds)
    elif args.direct:
        train_samples,test_samples = train_data,test_data

    train_dataset = data_utils.TensorDataset(torch.from_numpy(train_samples),torch.from_numpy(train_labels))
    train_sampler = create_weighted_sampler(train_labels)
    train_loader = data_utils.DataLoader(train_dataset,sampler=train_sampler,batch_size=args.batch_size)
    test_dataset = data_utils.TensorDataset(torch.from_numpy(test_samples),torch.from_numpy(test_labels))
    test_loader = data_utils.DataLoader(test_dataset,batch_size=1,shuffle=False)

    if args.dense:
        dense_layer(train_loader,test_loader,train_labels,test_labels,embed_dim,args.lr,args.output_path,args.epoch_num)
    elif args.random_forest:
        predictIBD(train_samples,train_labels,test_samples,test_labels)

main()

