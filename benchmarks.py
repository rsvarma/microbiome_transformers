##############################################################################
#Author: Rohan Varma
#
#Command Line tool for benchmarking results on AGP microbiome data.
#Allows testing of various sample aggregation strategies including multiplying 
#counts by an embedding matrix, averaging together embeddings, and
#weighted averaging of embeddings. Also allows to use a single dense layer
#or a random forest classifier.
###############################################################################

import argparse
import sys
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.optim import SGD
import sklearn
import numpy as np
import pdb
import tqdm
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
from inspect import signature
from transformers import AdamW,get_constant_schedule_with_warmup

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


def calc_aupr(y_true,y_probas,show_plot=False):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true,y_probas,pos_label = 1)
    aupr_score = metrics.auc(recall,precision)
    if show_plot:
        plt.figure()
        plt.plot(recall, precision)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Receiver operating characteristic')
        plt.show()    
    return aupr_score    



class Net(nn.Module):

    def __init__(self,input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim,1)

    def forward(self,x):
        x = torch.sigmoid(self.fc1(x))
        return x

class CENet(nn.Module):

    def __init__(self,input_dim):
        super(CENet, self).__init__()
        self.fc1 = nn.Linear(input_dim,2)
        self.softmax = nn.Softmax()

    def forward(self,x):
        x = self.softmax(self.fc1(x))
        return x        
        

def dense_layer(train_dataloader,train_orig_loader,test_dataloader,embed_dim,lr,log_file,epoch_num,cross_entropy=False,optim='sgd',betas=(0.9, 0.999), weight_decay = 0.01,warmup_steps=2000):
    with open(log_file,"w+") as f:
        f.write("EPOCH,MODE, AVG LOSS, TOTAL CORRECT, TOTAL ELEMENTS, ACCURACY, AUC, AUPR, TOTAL POSITIVE CORRECT, TOTAL POSITIVE, ACCURACY\n")

    if cross_entropy:
        net = CENet(embed_dim)
        criterion = nn.CrossEntropyLoss()
    else:
        net = Net(embed_dim)
        criterion = nn.MSELoss()
    optim_scheduler = None
    if optim == 'adam':
    # Setting the Adam optimizer with hyper-param
        print("USING ADAM OPTIMIZER, LR: {}".format(lr))
        optimizer = AdamW(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        optim_scheduler = get_constant_schedule_with_warmup(optimizer,warmup_steps)
    elif optim == "sgd":      
        print("USING SGD OPTIMIZER, LR: {}".format(lr))     
        optimizer = SGD(net.parameters(),lr=lr)
    pdb.set_trace()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        train_w_labels = []
        test_scores = []
        test_labels = []
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
            train_w_labels.append(labels)
            inputs,labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if cross_entropy:
                output = net(inputs.float())
                loss = criterion(output,labels.squeeze())
                predictions = output[:,1] >= 0.5
                train_scores.append(output[:,1].detach().cpu())
            else:
                output = net(inputs.float()).flatten()
                loss = criterion(output,labels.float())
                predictions = output >= 0.5
                train_scores.append(output.detach().cpu())
            loss.backward()
            optimizer.step()
            if optim_scheduler is not None:
                optim_scheduler.step()

            

            cumulative_loss += loss.item()

            total_train_correct += torch.sum(predictions == labels).item()
            total_train_samples += labels.shape[0]

            positive_inds = labels.nonzero(as_tuple=True)
            total_train_positive_correct += torch.sum(predictions[positive_inds] == labels[positive_inds]).item()
            total_train_positive += labels.nonzero().shape[0]
        auc_score = calc_auc(torch.cat(train_w_labels).numpy(),torch.cat(train_scores).numpy())
        aupr_score = calc_aupr(torch.cat(train_w_labels).numpy(),torch.cat(train_scores).numpy())
        with open(log_file,"a") as f:
            f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,"train",cumulative_loss/len(train_dataloader),total_train_correct,total_train_samples,total_train_correct/total_train_samples*100,auc_score,aupr_score,total_train_positive_correct,total_train_positive,total_train_positive_correct/total_train_positive*100))

        cumulative_loss = 0

        total_train_correct = 0
        total_train_samples = 0 
        total_train_positive_correct = 0
        total_train_positive = 0
        train_scores = []
        train_labels = []

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(train_orig_loader,
                              desc="EP_%s:%d" % ("train_orig", epoch),
                              total=len(train_orig_loader),
                              bar_format="{l_bar}{r_bar}")        
        #pdb.set_trace()
        net.eval()                              
        for inputs,labels in data_iter:
            train_labels.append(labels)
            inputs,labels = inputs.to(device), labels.to(device)
            if cross_entropy:
                output = net(inputs.float())
                loss = criterion(output,labels.squeeze())
                predictions = output[:,1] >= 0.5
                train_scores.append(output[:,1].detach().cpu())
            else:
                output = net(inputs.float()).flatten()
                loss = criterion(output,labels.float())
                predictions = output >= 0.5
                train_scores.append(output.detach().cpu())

            cumulative_loss += loss.item()

            total_train_correct += torch.sum(predictions == labels).item()
            total_train_samples += labels.shape[0]

            positive_inds = labels.nonzero(as_tuple=True)
            #pdb.set_trace()
            total_train_positive_correct += torch.sum(predictions[positive_inds] == labels[positive_inds]).item()
            total_train_positive += labels.nonzero().shape[0]

        auc_score = calc_auc(torch.cat(train_labels).numpy(),torch.cat(train_scores).numpy())
        aupr_score = calc_aupr(torch.cat(train_labels).numpy(),torch.cat(train_scores).numpy())
        with open(log_file,"a") as f:
            f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,"train_orig",cumulative_loss/len(train_dataloader),total_train_correct,total_train_samples,total_train_correct/total_train_samples*100,auc_score,aupr_score,total_train_positive_correct,total_train_positive,total_train_positive_correct/total_train_positive*100))

        cumulative_loss = 0
        total_test_samples = 0
        #pdb.set_trace()
        data_iter = tqdm.tqdm(test_dataloader,
                              desc="EP_%s:%d" % ("test", epoch),
                              total=len(test_dataloader),
                              bar_format="{l_bar}{r_bar}")       
        net.eval()   
        for inputs,labels in data_iter:
            test_labels.append(labels)
            inputs,labels = inputs.to(device), labels.to(device)
            if cross_entropy:
                output = net(inputs.float())
                loss = criterion(output,labels.squeeze())
                predictions = output[:,1] >= 0.5
                test_scores.append(output[:,1].detach().cpu())
            else:
                output = net(inputs.float()).flatten()
                loss = criterion(output,labels.float())
                predictions = output >= 0.5
                test_scores.append(output.detach().cpu())

                
            cumulative_loss += loss.item()

            total_test_samples += labels.shape[0]
            total_test_correct += torch.sum(predictions == labels).item()

            positive_inds = labels.nonzero(as_tuple=True)
            total_test_positive_correct += torch.sum(predictions[positive_inds] == labels[positive_inds]).item()
            total_test_positive += labels.nonzero().shape[0]
        auc_score = calc_auc(torch.cat(test_labels).numpy(),torch.cat(test_scores).numpy())
        aupr_score = calc_aupr(torch.cat(test_labels).numpy(),torch.cat(test_scores).numpy())         
        with open(log_file,"a") as f:
            f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,"test",cumulative_loss/len(test_dataloader),total_test_correct,total_test_samples,total_test_correct/total_test_samples*100,auc_score,aupr_score,total_test_positive_correct,total_test_positive,total_test_positive_correct/total_test_positive*100))


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
    aupr_score = metrics.auc(recall,precision)

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
    
    return(roc_auc,aupr_score, fpr, tpr, average_precision, f1, f2)



def predictIBD(X_train, y_train, X_test, y_test, graph_title = "IBD Tests", max_depth = 12, n_estimators = 140, plot = True, plot_pr = True, weight = 20, feat_imp = False, flipped = False):
    weights = {0:1, 1:weight}
    m = RandomForestClassifier(max_depth= max_depth, random_state=0, n_estimators= n_estimators, class_weight = weights)
    m.fit(X_train, y_train)
    probs = m.predict_proba(X_test)
    probs_train = m.predict_proba(X_train)
    roc_auc,aupr, fpr, tpr, precision, f1, f2 = computeMLstats(m, data = X_test, y = y_test, plot = plot, plot_pr = plot_pr, graph_title = "IBD test", flipped = flipped)
    roc_auc,aupr, fpr, tpr, precision, f1, f2 = computeMLstats(m, data = X_train, y = y_train, plot = plot, plot_pr = plot_pr, graph_title = "IBD train", flipped = flipped)
    plt.show()

    #feat_imp_sort = getFeatureImportance(m, data = X_train, y = y_train)
    
    return(m, roc_auc, None, fpr, tpr, precision, f1, f2)#, feat_imp_sort)



def embed_matrix(train_freqs,embeds):
    train_freqs = np.arcsinh(train_freqs)
    #train_samples = (np.matmul(train_freqs,embeds))   
    train_samples = preprocessing.scale(np.matmul(train_freqs,embeds))
    #test_samples = preprocessing.scale(np.matmul(test_freqs,embeds))

    return train_samples


def average(train_embeddings):
    train_samples = preprocessing.scale(np.average(train_embeddings,axis=1))
    return train_samples

def weighted_average(train_embeddings,train_freqs):
    train_weights = train_freqs/train_freqs.sum(1)[:,None]
    train_samples = np.zeros((train_embeddings.shape[0],train_embeddings.shape[2]))
    for i in range(train_embeddings.shape[0]):
        train_samples[i] = np.average(train_embeddings[i],axis=0,weights=train_weights[i])    
    return train_samples

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--dataset", required=True, type=str, help="path to numpy array for asv counts")
    parser.add_argument("-ep", "--embedding_path", required=True, type=str, help="path to numpy array for GloVE embeddings")
    parser.add_argument("-l","--labels", required=True,type=str,help="path to numpy array for labels")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="path to output log file")
    parser.add_argument("-b", "--batch_size", type=int, default=48, help="batch size for training dense layer")
    parser.add_argument("-e", "--epoch_num", type = int, default=20, help="number of epochs to train dense layer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of dense layer")

    parser.add_argument("--test_samples",required=False,type=str,help="microbiome samples for test set, only provide if not trying to do cross validation. If providing this, --samples should provide the path to the training samples")
    parser.add_argument("--test_labels",required=False,type=str,help="labels for test set, only provide if not wanting to perform cross validation. If provided --sample_labels should provide the path to the train labels")

    parser.add_argument("--ce",dest='cross_entropy',action='store_true',help="train with cross entropy loss")
    parser.add_argument("--mse",dest='cross_entropy',action='store_false',help="train with mse loss")
    parser.set_defaults(cross_entropy=False)


    parser.add_argument("--adam",dest='optim',action='store_const',const='adam',help="train with adam optimizer")
    parser.add_argument("--sgd",dest='optim',action='store_const',const='sgd',help="train with sgd")
    parser.set_defaults(optim='sgd')

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


    data = np.load(args.dataset)
    labels = np.load(args.labels).flatten()
    embeds = np.load(args.embedding_path)
    embed_dim = embeds.shape[1]


    if not args.direct:
        embeddings = np.zeros((data.shape[0],embeds.shape[0],embed_dim))
        freqs = np.zeros((data.shape[0],embeds.shape[0]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i,j,1] > 0:
                    embeddings[i,int(data[i,j,0])] = embeds[int(data[i,j,0])]
                    freqs[i,int(data[i,j,0])] = data[i,j,1]

    if args.average:
        samples = average(embeddings)
    elif args.waverage:
        samples = weighted_average(embeddings,freqs)
    elif args.ematrix:
        samples= embed_matrix(freqs, embeds)
    elif args.direct:
        samples = data

    if args.test_samples and args.test_labels is not None: 
        log_file = args.output_path+".txt"
        test_data = np.load(args.test_samples)
        test_labels = np.load(args.test_labels)

        if not args.direct:
            test_embeddings = np.zeros((test_data.shape[0],embeds.shape[0],embed_dim))
            test_freqs = np.zeros((test_data.shape[0],embeds.shape[0]))
            for i in range(test_data.shape[0]):
                for j in range(test_data.shape[1]):
                    if test_data[i,j,1] > 0:
                        test_embeddings[i,int(test_data[i,j,0])] = embeds[int(test_data[i,j,0])]
                        test_freqs[i,int(test_data[i,j,0])] = test_data[i,j,1]        

        if args.average:
            test_samples = average(test_embeddings)
        elif args.waverage:
            test_samples = weighted_average(test_embeddings,freqs)
        elif args.ematrix:
            test_samples= embed_matrix(test_freqs, embeds)
        elif args.direct:
            test_samples = test_data

        train_samples = samples
        train_labels = labels

        train_dataset = data_utils.TensorDataset(torch.from_numpy(train_samples),torch.from_numpy(train_labels))
        train_sampler = create_weighted_sampler(train_labels)
        train_loader = data_utils.DataLoader(train_dataset,sampler=train_sampler,batch_size=args.batch_size)
        train_orig_loader = data_utils.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False)
        test_dataset = data_utils.TensorDataset(torch.from_numpy(test_samples),torch.from_numpy(test_labels))
        test_loader = data_utils.DataLoader(test_dataset,batch_size=35,shuffle=False)
        if args.dense:
            dense_layer(train_loader,train_orig_loader,test_loader,embed_dim,args.lr,log_file,args.epoch_num,args.cross_entropy,args.optim)
        elif args.random_forest:
            predictIBD(train_samples,train_labels,test_samples,test_labels)


    elif args.test_samples is None and args.test_labels is None:


        split_count = 1
        kf = KFold(n_splits=5,shuffle=True,random_state=42)
        for train_index,test_index in kf.split(samples):
            log_file = args.output_path+"_valset"+str(split_count)+".txt"
            train_samples = samples[train_index]
            train_labels = labels[train_index]
            test_samples = samples[test_index]
            test_labels = labels[test_index]
            train_dataset = data_utils.TensorDataset(torch.from_numpy(train_samples),torch.from_numpy(train_labels))
            train_sampler = create_weighted_sampler(train_labels)
            train_loader = data_utils.DataLoader(train_dataset,sampler=train_sampler,batch_size=args.batch_size)
            train_orig_loader = data_utils.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False)
            test_dataset = data_utils.TensorDataset(torch.from_numpy(test_samples),torch.from_numpy(test_labels))
            test_loader = data_utils.DataLoader(test_dataset,batch_size=35,shuffle=False)

            if args.dense:
                dense_layer(train_loader,train_orig_loader,test_loader,embed_dim,args.lr,log_file,args.epoch_num,args.cross_entropy,args.optim)
            elif args.random_forest:
                predictIBD(train_samples,train_labels,test_samples,test_labels)
            split_count += 1

main()


