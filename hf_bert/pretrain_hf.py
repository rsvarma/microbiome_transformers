import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader



from transformers import BertConfig,BertForMaskedLM
import tqdm
import pdb

class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BertForMaskedLM, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 100, log_file=None,training_checkpoint=None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.hardware = "cuda" if cuda_condition else "cpu"

        # This BERT model will be saved every epoch
        self.bert = bert.to(self.device)
        self.bert = self.bert.float()

        #pdb.set_trace()
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.bert = nn.DataParallel(self.bert, device_ids=cuda_devices)
            self.hardware = "parallel"
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        #self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        #self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)
        self.optim = SGD(self.bert.parameters(),lr=lr,momentum=0.9)

        self.log_freq = log_freq

        # clear log file
        if log_file:
            self.log_file = log_file
            if(training_checkpoint is None):
                with open(self.log_file,"w+") as f:
                    f.write("EPOCH,MODE,TOTAL CORRECT,AVG LOSS,TOTAL ELEMENTS,ACCURACY,MASK CORRECT,TOTAL MASK,MASK ACCURACY\n")
        print("Total Parameters:", sum([p.nelement() for p in self.bert.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"



        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        cumulative_loss = 0.0

        total_correct = 0
        total_element = 0
        total_mask_correct = 0
        total_mask = 0
        for i, data in data_iter:
            #pdb.set_trace()
            #print(i)
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            #create attention mask
            zero_boolean = torch.eq(data["bert_input"],0).all(2)
            mask = torch.ones(zero_boolean.shape,dtype=torch.float).to(self.device)
            mask = mask.masked_fill(zero_boolean,0)

            #create labels for huggingface BERT model (tensor of shape (batch size, sequence len) where labels for all non mask tokens are -100)
            mask_labels = data["bert_label"].clone()
            mask_labels = mask_labels.masked_fill(~data["mask_locations"],-100)

            # 1. forward the next_sentence_prediction and masked_lm model
            mask_loss,mask_scores = self.bert.forward(inputs_embeds=data["bert_input"],attention_mask=mask,labels=mask_labels)

            # 3. backward and optimization only in train
            if train:
                #self.optim_schedule.zero_grad()
                self.optim.zero_grad()
                if self.hardware == "parallel":
                    mask_loss.sum().backward()
                else:
                    mask_loss.backward()
                #self.optim_schedule.step_and_update_lr()
                self.optim.step()

            mask_predictions = torch.masked_select(mask_scores.max(2).indices,data["mask_locations"])
            mask_token_labels = torch.masked_select(data["bert_label"],data["mask_locations"])
            total_mask_correct += torch.sum(mask_predictions == mask_token_labels).item()
            total_mask += mask_token_labels.shape[0]            
            
            #get accuracy for all tokens
            total_correct += torch.sum(mask_scores.max(2).indices == data["bert_label"]).item()
            total_element += data["bert_input"].shape[0]*data["bert_input"].shape[1]

            log_loss = 0
            if self.hardware == "parallel":
                cumulative_loss += mask_loss.sum().item()
                log_loss = mask_loss.sum().item()

            else:
                cumulative_loss += mask_loss.item()        
                log_loss = mask_loss.item()    
            if i % self.log_freq == 0:
                data_iter.write("epoch: {}, iter: {}, avg loss: {}, mask accuracy: {}/{}={:.2f}%, loss: {}".format(epoch,i,cumulative_loss/(i+1),total_mask_correct,total_mask,total_mask_correct/total_mask*100,log_loss))

 
            del data
            del mask
            del mask_labels
            del mask_loss
            del mask_scores
            del mask_predictions
            del mask_token_labels

        print("EP{}_{}, avg_loss={}, accuracy={:.2f}%".format(epoch,str_code,cumulative_loss / len(data_iter),total_mask_correct/total_mask*100))
        if self.log_file:
            with open(self.log_file,"a") as f:
                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,str_code,cumulative_loss/len(data_iter),total_correct,total_element,total_correct/total_element*100,total_mask_correct,total_mask,total_mask_correct/total_mask*100))

 
        
    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_bert_path = "{}_BERT.ep{}".format(file_path,epoch)
        output_LM_path = "{}_LM.ep{}".format(file_path,epoch)
        torch.save(self.bert.state_dict(), output_bert_path)
        if self.hardware == "parallel":
            torch.save(self.model.module.state_dict(),output_LM_path)
        else:
            torch.save(self.model.state_dict(),output_LM_path)
        print("EP:%d BERT Model Saved on:" % epoch,output_bert_path)