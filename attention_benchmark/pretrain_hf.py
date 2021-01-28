import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader
from sklearn import metrics


from electra_discriminator import AttentionModel
import tqdm
import pdb

class ELECTRATrainer:
    """
    ELECTRATrainer make the pretrained ELECTRA model 
    """

    def __init__(self, electra: AttentionModel, vocab_size: int,
                 train_dataloader: DataLoader, train_orig_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 100, log_file=None,
                 freeze_embed=0,class_weights=None):
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
        self.loss = nn.MSELoss()

        self.loss.to(self.device)

        #pdb.set_trace()
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for ELECTRA" % torch.cuda.device_count())
            self.electra = nn.DataParallel(self.electra, device_ids=cuda_devices)
            self.hardware = "parallel"
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.train_orig_data = train_orig_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        #self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        #self.optim_schedule = ScheduledOptim(self.optim, self.electra.hidden, n_warmup_steps=warmup_steps)    
        if freeze_embed == 1:
            if self.hardware == "parallel":
                self.electra.module.embed_layer.weight.requires_grad = False
            else:
                self.electra.embed_layer.weight.requires_grad = False
        elif freeze_embed == 2:
            self.freeze_embed_idx = torch.arange(26726,dtype=torch.long).to(self.device)
        self.optim = SGD([param for param in self.electra.parameters() if param.requires_grad == True],lr=lr,momentum=0.9)

        self.log_freq = log_freq

        # clear log file
        if log_file:
            self.log_file = log_file
            with open(self.log_file,"w+") as f:
                f.write("EPOCH,MODE, AVG LOSS, TOTAL CORRECT, TOTAL ELEMENTS, ACCURACY, AUC, AUPR, TOTAL POSITIVE CORRECT, TOTAL POSITIVE, ACCURACY\n")
        print("Total Parameters:", sum([p.nelement() for p in self.electra.parameters()]))

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

    def train(self, epoch):
        self.electra.train()
        self.iteration(epoch, self.train_data,True,"train")

    def train_orig_dist(self,epoch):
        self.electra.eval()
        self.iteration(epoch,self.train_orig_data,False,"train_orig")

    def test(self, epoch):
        self.electra.eval()
        self.iteration(epoch, self.test_data,False,"test")

    def iteration(self, epoch, data_loader, train,str_code):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        



        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        cumulative_loss = 0.0

        total_correct = 0
        total_samples = 0
        total_positive_correct = 0
        total_positive = 0
        all_scores = []
        all_labels = []
        for i, data in data_iter:
            #pdb.set_trace()
            #print(i)
            
            all_labels.append(data["electra_label"])
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            #create attention mask
            #pdb.set_trace()
            zero_boolean = torch.eq(data["species_frequencies"],0)
            mask = torch.ones(zero_boolean.shape,dtype=torch.float).to(self.device)
            mask = mask.masked_fill(zero_boolean,0)

            
            # 1. forward the next_sentence_prediction and masked_lm model
            scores = self.electra.forward(data["electra_input"],mask)              
            # 3. backward and optimization only in train
            loss = self.loss(scores,data["electra_label"].float())
            if train:
                #self.optim_schedule.zero_grad()
                #pdb.set_trace()
                self.optim.zero_grad()
                loss.backward()
                #self.optim_schedule.step_and_update_lr()
                if self.freeze_embed == 2:
                    if self.hardware == "parallel":
                        self.electra.module.embed_layer.weight.grad[self.freeze_embed_idx] = 0
                    else:
                        self.electra.embed_layer.weight.grad[self.freeze_embed_idx] = 0
                self.optim.step()

            all_scores.append(scores.detach().cpu())
            predictions = scores >= 0.5
       
            
            #get accuracy for all tokens
            total_correct += torch.sum(predictions == data["electra_label"]).item()
            total_samples += data["electra_input"].shape[0]

            positive_inds = data["electra_label"].nonzero(as_tuple=True)
            total_positive_correct += torch.sum(predictions[positive_inds] == data["electra_label"][positive_inds]).item()
            total_positive += data["electra_label"].nonzero().shape[0]

            log_loss = 0
            if self.hardware == "parallel":
                cumulative_loss += loss.sum().item()
                log_loss = loss.sum().item()

            else:
                cumulative_loss += loss.item()        
                log_loss = loss.item()    
            if i % self.log_freq == 0:
                if total_positive > 0:
                    data_iter.write("epoch: {}, iter: {}, avg loss: {},accuracy: {}/{}={:.2f}%,pos accuracy: {}/{}={:.2f}%, loss: {}".format(epoch,i,cumulative_loss/(i+1),total_correct,total_samples,total_correct/total_samples*100,total_positive_correct,total_positive,total_positive_correct/total_positive*100,log_loss))
                else:
                    data_iter.write("epoch: {}, iter: {}, avg loss: {},accuracy: {}/{}={:.2f}%,pos accuracy: 0/0, loss: {}".format(epoch,i,cumulative_loss/(i+1),total_correct,total_samples,total_correct/total_samples*100,log_loss))                    
 
            del data
            del mask
            del loss
            del scores
            del predictions
            del positive_inds

  
        auc_score = ELECTRATrainer.calc_auc(torch.cat(all_labels).flatten().numpy(),torch.cat(all_scores).flatten().numpy())
        aupr_score = ELECTRATrainer.calc_aupr(torch.cat(all_labels).flatten().numpy(),torch.cat(all_scores).flatten().numpy())
        print("EP{}_{}, avg_loss={}, accuracy={:.2f}%".format(epoch,str_code,cumulative_loss / len(data_iter),total_correct/total_samples*100))
        if self.log_file:
            with open(self.log_file,"a") as f:
                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,str_code,cumulative_loss/len(data_iter),total_correct,total_samples,total_correct/total_samples*100,auc_score,aupr_score,total_positive_correct,total_positive,total_positive_correct/total_positive*100))

 
        
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