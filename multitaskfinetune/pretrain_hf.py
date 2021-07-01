import torch
import torch.nn as nn
from transformers import AdamW,get_constant_schedule_with_warmup
from torch.optim import SGD
from torch.utils.data import DataLoader
from sklearn import metrics
import math

import os
from electra_discriminator import ElectraDiscriminator
from transformers import ElectraConfig,ElectraForSequenceClassification
import tqdm
import pdb

class ELECTRATrainer:
    """
    ELECTRATrainer make the pretrained ELECTRA model 
    """

    def __init__(self, electra: ElectraDiscriminator, vocab_size: int,
                 train_dataloaders: DataLoader, train_orig_dataloader: DataLoader, task_log_files,log_file,test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=2000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 100,
                 freeze_embed=0,class_weights=None,loss_func='mse',optim='sgd',hidden_size=None):
        """
        :param electra: ELECTRA model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        self.optim = optim
        self.loss_func = loss_func
        self.softmax = torch.nn.Softmax()
        # Setup cuda device for ELECTRA training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.hardware = "cuda" if cuda_condition else "cpu"

        self.freeze_embed = freeze_embed

        self.electra = electra
        self.electra = self.electra.to(self.device)
        self.electra = self.electra.float()
        #pdb.set_trace()
        #for cross entropy loss
        if self.loss_func == 'ce':
            if class_weights is not None:
                class_weights.to(self.device)
                self.loss = nn.CrossEntropyLoss(class_weights)
            else:
                print("USING CROSS ENTROPY LOSS")
                self.loss = nn.CrossEntropyLoss()
        elif self.loss_func == 'mse':
            print("USING MEAN SQUARED ERROR LOSS")
            self.loss = nn.MSELoss()
        self.loss.to(self.device)

        #pdb.set_trace()
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for ELECTRA" % torch.cuda.device_count())
            self.electra = nn.DataParallel(self.electra, device_ids=cuda_devices)
            self.hardware = "parallel"
        
        # Setting the train and test data loader
        self.train_data = train_dataloaders
        self.train_orig_data = train_orig_dataloader
        self.test_data = test_dataloader


        if freeze_embed == 1:
            if self.hardware == "parallel":
                self.electra.module.embed_layer.weight.requires_grad = False
            else:
                self.electra.embed_layer.weight.requires_grad = False
        elif freeze_embed == 2:
            self.freeze_embed_idx = torch.arange(26726,dtype=torch.long).to(self.device)

        self.scheduler = None
        if self.optim == 'adam':
        # Setting the Adam optimizer with hyper-param
            print("USING ADAM OPTIMIZER, LR: {}".format(lr))
            self.optim = AdamW([param for param in self.electra.parameters() if param.requires_grad == True], lr=lr, betas=betas, weight_decay=weight_decay)
            self.scheduler = get_constant_schedule_with_warmup(self.optim,warmup_steps,)               
        elif self.optim == 'sgd':
            print("USING SGD OPTIMIZER, LR: {}".format(lr))
            self.optim = SGD([param for param in self.electra.parameters() if param.requires_grad == True],lr=lr,momentum=0.9)

        self.log_freq = log_freq

        # clear log file

        self.task_log_files = task_log_files
        self.log_file = log_file
        self.num_tasks = len(task_log_files)
        for file in self.task_log_files:
            with open(file,"w+") as f:
                f.write("EPOCH,MODE, AVG LOSS, POSITIVE LOSS, NEGATIVE LOSS, TOTAL CORRECT, TOTAL ELEMENTS, ACCURACY, AUC, AUPR, TOTAL POSITIVE CORRECT, TOTAL POSITIVE, ACCURACY\n")
        print("Total Parameters:", sum([p.nelement() for p in self.electra.parameters()]))
        with open(self.log_file,"w+") as f:
            f.write("EPOCH, MODE, AVG LOSS\n")
    @staticmethod
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

    @staticmethod
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

    def train(self, epoch,multi):
        self.electra.train()
        self.iteration(epoch, self.train_data,True,"train",multi)

    def train_orig_dist(self,epoch,multi):
        self.electra.eval()
        self.iteration(epoch,self.train_orig_data,False,"train_orig",multi)

    def test(self, epoch,multi):
        self.electra.eval()
        self.iteration(epoch, self.test_data,False,"test",multi)

    def iteration(self, epoch, data_loader, train,str_code,multi=False):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        

        if isinstance(data_loader, list):
            iter = zip(*data_loader)
            loader_lens = []
            for loader in data_loader:
                loader_lens.append(len(loader))
            set_length = min(loader_lens)
        else:
            iter = data_loader
            set_length = len(data_loader)

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(iter),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=set_length,
                              bar_format="{l_bar}{r_bar}")

        cumulative_loss = 0.0

        total_correct = [0 for i in range(self.num_tasks)]
        total_samples = [0 for i in range(self.num_tasks)]
        total_positive_correct = [0 for i in range(self.num_tasks)]
        total_positive = [0 for i in range(self.num_tasks)]
        all_scores = [[] for i in range(self.num_tasks)]
        all_labels = [[] for i in range(self.num_tasks)]
        cumulative_task_losses = [0 for i in range(self.num_tasks)]
        #losses on positive examples for each task
        cumulative_task_pos_losses = [0 for i in range(self.num_tasks)]
        #losses on negative examples for each task
        cumulative_task_neg_losses = [0 for i in range(self.num_tasks)]
        batch_num = 0
        for i, data in data_iter:
            batch_num += 1
            #print(i)
            # 0. batch_data will be sent into the device(GPU or cpu)
            if isinstance(data,tuple):
                merged_data = data[0]
                for i in range(1,len(data)):
                    merged_data = {"electra_input": torch.vstack((merged_data['electra_input'],data[i]['electra_input'])),
                                 "electra_label": torch.vstack((merged_data['electra_label'],data[i]['electra_label'])),
                                "species_frequencies": torch.vstack((merged_data['species_frequencies'],data[i]['species_frequencies'])),
                                }
                data = merged_data
            data = {key: value.to(self.device) for key, value in data.items()}
            #create attention mask
            #pdb.set_trace()
            zero_boolean = torch.eq(data["species_frequencies"],0)
            mask = torch.ones(zero_boolean.shape,dtype=torch.float).to(self.device)
            mask = mask.masked_fill(zero_boolean,0)

            
            # 1. forward the next_sentence_prediction and masked_lm model
            scores = self.electra.forward(data["electra_input"],mask)            
            # 3. backward and optimization only in train

            #for mse loss
            #print("batch {} losses: ".format(batch_num)) 
            if self.loss_func == 'mse':
                loss = 0
                for i,task_scores in enumerate(scores):
                    lab_mask = data["electra_label"][:,i] != -100             
                    t_loss = self.loss(task_scores[lab_mask,1],data["electra_label"][lab_mask,i].float())
                    #pdb.set_trace()
                    if math.isnan(t_loss.item()):
                        continue
                    pos_mask = data["electra_label"][:,i] == 1
                    neg_mask = data["electra_label"][:,i] == 0
                    pos_loss = self.loss(task_scores[pos_mask,1],data["electra_label"][pos_mask,i].float())
                    neg_loss = self.loss(task_scores[neg_mask,1],data["electra_label"][neg_mask,i].float())
                    if not math.isnan(pos_loss.item()):
                        cumulative_task_pos_losses[i] += pos_loss.item()*pos_mask.sum().item()
                    if not math.isnan(neg_loss.item()):
                        cumulative_task_neg_losses[i] += neg_loss.item()*neg_mask.sum().item()
                    loss += t_loss
                    cumulative_task_losses[i] += t_loss.item()*lab_mask.sum().item()
                    #print(t_loss)

            #for cross entropy
            elif self.loss_func == 'ce':
                loss = 0
                for i,task_scores in enumerate(scores):
                    #pdb.set_trace()
                    lab_mask = data["electra_label"][:,i] != -100     
                    if lab_mask.sum().item() == 0:
                        continue                
                    pos_mask = data["electra_label"][:,i] == 1
                    neg_mask = data["electra_label"][:,i] == 0
                    if pos_mask.sum().item() > 0:
                        pos_loss = self.loss(task_scores[pos_mask],data["electra_label"][pos_mask,i])
                        cumulative_task_pos_losses[i] += pos_loss.item()*pos_mask.sum().item()
                    if neg_mask.sum().item()> 0:
                        neg_loss = self.loss(task_scores[neg_mask],data["electra_label"][neg_mask,i])                        
                        cumulative_task_neg_losses[i] += neg_loss.item()*neg_mask.sum().item()

                    t_loss = self.loss(task_scores,data["electra_label"][:,i])
                    loss += t_loss
                    cumulative_task_losses[i] += t_loss.item()*lab_mask.sum().item()
            if train:
                self.optim.zero_grad()
                loss.backward()
                if self.freeze_embed == 2:
                    if self.hardware == "parallel":
                        self.electra.module.embed_layer.weight.grad[self.freeze_embed_idx] = 0
                    else:
                        self.electra.embed_layer.weight.grad[self.freeze_embed_idx] = 0                      
                self.optim.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            for i,task_scores in enumerate(scores):
                task_label = data["electra_label"][:,i]
                sample_mask = task_label != -100
                task_label = task_label[sample_mask]
                all_labels[i].append(task_label.detach().cpu())
                all_scores[i].append(task_scores[sample_mask,1].detach().cpu())
                #for mse loss
                if self.loss_func == 'mse':
                    predictions = task_scores[:,1] >= 0.5
                    predictions = predictions[sample_mask]

                #for cross entropy
                elif self.loss_func == 'ce':    
                    predictions = task_scores.max(1).indices
                    predictions = predictions[sample_mask]
        
                
                #get accuracy for all tokens
                total_correct[i] += torch.sum(predictions == task_label).item()
                total_samples[i] += predictions.shape[0]

                positive_inds = task_label.nonzero(as_tuple=True)
                total_positive_correct[i] += torch.sum(predictions[positive_inds] == task_label[positive_inds]).item()
                total_positive[i] += task_label.nonzero().shape[0]

            log_loss = 0
            if self.hardware == "parallel":              
                cumulative_loss += loss.sum().item()
                log_loss = loss.sum().item()

            else:
                cumulative_loss += loss.item()        
                log_loss = loss.item() 
            #print("batch {} cumulative loss: {}".format(batch_num,cumulative_loss))                   
 
            del data
            del mask
            del loss
            del scores
            del predictions
            del positive_inds
        #pdb.set_trace()
        for i in range(self.num_tasks):
            auc_score = 0
            aupr_score = 0
            auc_score = ELECTRATrainer.calc_auc(torch.cat(all_labels[i]).flatten().numpy(),torch.cat(all_scores[i]).numpy())
            aupr_score = ELECTRATrainer.calc_aupr(torch.cat(all_labels[i]).flatten().numpy(),torch.cat(all_scores[i]).numpy())       
            with open(self.task_log_files[i],"a") as f:
                f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,str_code,cumulative_task_losses[i]/total_samples[i],cumulative_task_pos_losses[i]/total_positive[i],cumulative_task_neg_losses[i]/(total_samples[i]-total_positive[i]),total_correct[i],total_samples[i],total_correct[i]/total_samples[i]*100,auc_score,aupr_score,total_positive_correct[i],total_positive[i],total_positive_correct[i]/total_positive[i]*100))
        with open(self.log_file,"a") as f:
            f.write("{},{},{}\n".format(epoch,str_code,cumulative_loss/len(data_iter)))
 
        
    def save(self, epoch, file_path):
        """
        Saving the current ELECTRA model on file_path

        :param epoch: current epoch number
        :param file_path: model output directory
        """
        output_file_path = file_path+"_epoch{}".format(epoch)
        if self.hardware == "parallel":
            #pdb.set_trace()
            self.electra.module.discriminator.save_pretrained(output_file_path+"_disc")
            torch.save(self.electra.module.embed_layer.state_dict(),output_file_path+"_embed")
        else:
            self.electra.discriminator.save_pretrained(output_file_path+"_disc")
            torch.save(self.electra.embed_layer.state_dict(),output_file_path+"_embed")