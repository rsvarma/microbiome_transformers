import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader


from electra_pretrain_model import ElectraPretrainModel
import tqdm
import pdb

class ELECTRATrainer:
    """
    ELECTRATrainer make the pretrained ELECTRA model 
    """

    def __init__(self, electra: ElectraPretrainModel, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 100, log_file=None,training_checkpoint=None):
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
        # Setup cuda device for ELECTRA training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.hardware = "cuda" if cuda_condition else "cpu"

        # This ELECTRA model will be saved every epoch
        self.electra = electra.to(self.device)
        self.electra = self.electra.float()

        #pdb.set_trace()
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for ELECTRA" % torch.cuda.device_count())
            self.electra = nn.DataParallel(self.electra, device_ids=cuda_devices)
            self.hardware = "parallel"
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        #self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        #self.optim_schedule = ScheduledOptim(self.optim, self.electra.hidden, n_warmup_steps=warmup_steps)
        self.optim = SGD(self.electra.parameters(),lr=lr,momentum=0.9)

        self.log_freq = log_freq

        # clear log file
        if log_file:
            self.log_file = log_file
            if(training_checkpoint is None):
                with open(self.log_file,"w+") as f:
                    f.write("EPOCH,MODE,AVG LOSS,TOTAL CORRECT,TOTAL ELEMENTS,ACCURACY,MASK CORRECT,TOTAL MASK,MASK ACCURACY,GEN TOTAL ACCURACY, GEN MASK ACCURACY\n")
        print("Total Parameters:", sum([p.nelement() for p in self.electra.parameters()]))

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

        g_total_correct = 0
        g_total_mask_correct = 0
        d_total_correct = 0
        total_element = 0
        d_total_mask_correct = 0
        total_mask = 0
        for i, data in data_iter:
 
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            #create attention mask
            zero_boolean = torch.eq(data["species_frequencies"],0)
            mask = torch.ones(zero_boolean.shape,dtype=torch.float).to(self.device)
            mask = mask.masked_fill(zero_boolean,0)

            #change label for non-masked tokens to -100 so generator ignores predictions on non-masked tokens
            data["electra_mask_label"] = data["electra_label"].masked_fill(~data["mask_locations"],-100) 
            
            loss,d_scores,g_scores,d_labels = self.electra.forward(data,mask)             
            # 3. backward and optimization only in train
            if train:
                #self.optim_schedule.zero_grad()
                self.optim.zero_grad()
                if self.hardware == "parallel":
                    #pdb.set_trace()
                    loss.mean().backward()
                else:
                    loss.backward()
                #self.optim_schedule.step_and_update_lr()
                self.optim.step()
            #pdb.set_trace()

            #get generator accuracy for masked tokens
            g_predictions = g_scores.max(2)[1]
            g_mask_predictions = torch.masked_select(g_predictions,data["mask_locations"])
            g_mask_token_labels = torch.masked_select(data["electra_mask_label"],data["mask_locations"])
            g_total_mask_correct += torch.sum(g_mask_predictions == g_mask_token_labels).item()


            del g_mask_predictions
            del g_mask_token_labels

            #get generator accuracy for all tokens
            g_total_correct += torch.masked_select((g_predictions == data["electra_label"]),mask.bool()).sum().item()

            del g_predictions
            del g_scores

            #get discriminator accuracy for all tokens
            d_predictions = torch.where(d_scores > 0.5,torch.tensor([1]).to(self.device),torch.tensor([0]).to(self.device))
            d_mask_predictions = torch.masked_select(d_predictions,data["mask_locations"])
            d_mask_token_labels = torch.masked_select(d_labels,data["mask_locations"])
            d_total_mask_correct += torch.sum(d_mask_predictions == d_mask_token_labels).item()
            total_mask += d_mask_token_labels.shape[0]     

            del d_mask_predictions
            del d_mask_token_labels       
            
            #get discriminator accuracy for all tokens
            d_total_correct += torch.masked_select((d_predictions == d_labels),mask.bool()).sum().item()
            total_element += mask.sum().item()


            log_loss = 0
            if self.hardware == "parallel":
                cumulative_loss += loss.sum().item()
                log_loss = loss.sum().item()

            else:
                cumulative_loss += loss.item()        
                log_loss = loss.item()    
            if i % self.log_freq == 0:
                data_iter.write("epoch: {}, iter: {}, avg loss: {},accuracy: {}/{}={:.2f}%, mask accuracy: {}/{}={:.2f}%, loss: {}".format(epoch,i,cumulative_loss/(i+1),d_total_correct,total_element,d_total_correct/total_element*100,d_total_mask_correct,total_mask,d_total_mask_correct/total_mask*100,log_loss))

 
            del data
            del mask
            del loss
            del d_scores
            del d_labels
            del d_predictions


        print("EP{}_{}, avg_loss={}, accuracy={:.2f}%".format(epoch,str_code,cumulative_loss / len(data_iter),d_total_mask_correct/total_mask*100))
        if self.log_file:
            with open(self.log_file,"a") as f:
                f.write("{},{},{},{},{},{:4f},{},{},{:4f},{:4f},{:4f}\n".format(epoch,str_code,cumulative_loss/len(data_iter),d_total_correct,total_element,d_total_correct/total_element*100,d_total_mask_correct,total_mask,d_total_mask_correct/total_mask*100,g_total_correct/total_element*100,g_total_mask_correct/total_mask*100))

 
        
    def save(self, epoch, file_path):
        """
        Saving the current ELECTRA model on file_path

        :param epoch: current epoch number
        :param file_path: model output directory
        """
        output_file_path = file_path+"_epoch{}".format(epoch)
        if self.hardware == "parallel":
            #pdb.set_trace()
            self.electra.module.discriminator.discriminator.save_pretrained(output_file_path+"_disc")
            torch.save(self.electra.module.discriminator.embed_layer.state_dict(),output_file_path+"_embed")
        else:
            self.electra.discriminator.discriminator.save_pretrained(output_file_path+"_disc")
            torch.save(self.electra.discriminator.embed_layer.state_dict(),output_file_path+"_embed")