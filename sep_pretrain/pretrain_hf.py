import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader


from electra_pretrain_model import ElectraGenerator, ElectraDiscriminator
import tqdm
import pdb

class ELECTRATrainer:
    """
    ELECTRATrainer make the pretrained ELECTRA model 
    """

    def __init__(self, electra_gen: ElectraGenerator, electra_disc: ElectraDiscriminator, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 100, g_log_file=None,d_log_file = None,append=False):
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
        self.electra_gen = electra_gen.to(self.device)
        self.electra_gen = self.electra_gen.float()

        self.electra_disc = electra_disc.to(self.device)
        self.electra_disc = self.electra_disc.float()

        #pdb.set_trace()
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for ELECTRA" % torch.cuda.device_count())
            self.electra_gen = nn.DataParallel(self.electra_gen, device_ids=cuda_devices)
            self.electra_disc = nn.DataParallel(self.electra_disc, device_ids=cuda_devices)
            self.hardware = "parallel"
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        #self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        #self.optim_schedule = ScheduledOptim(self.optim, self.electra.hidden, n_warmup_steps=warmup_steps)
        #self.optimg = Adam(self.electra_gen.parameters(),lr=lr,eps=1e-06)
        #self.optimd = Adam(self.electra_disc.parameters(),lr=lr,eps=1e-06)
        self.optimg = SGD(self.electra_gen.parameters(),lr=lr,momentum=0.9)
        self.optimd = SGD(self.electra_disc.parameters(),lr=lr,momentum=0.9)

        self.log_freq = log_freq

        # clear log file
        if d_log_file:
            self.d_log_file = d_log_file
            if not append:
                with open(self.d_log_file,"w+") as f:
                    f.write("EPOCH,MODE,AVG LOSS,TOTAL CORRECT,TOTAL ELEMENTS,ACCURACY,MASK CORRECT,TOTAL MASK,MASK ACCURACY\n")

        # clear log file
        if g_log_file:
            self.g_log_file = g_log_file
            if not append:
                with open(self.g_log_file,"w+") as f:
                    f.write("EPOCH,MODE,AVG LOSS,TOTAL CORRECT,TOTAL ELEMENTS,ACCURACY,MASK CORRECT,MASK 3 CORRECT,MASK 5 CORRECT,MASK 10 CORRECT,TOTAL MASK,MASK ACCURACY\n")

    def train(self, epoch,disc=False):
        self.iteration(epoch, self.train_data,train=True,disc=disc)

    def test(self, epoch,disc=False):
        self.iteration(epoch, self.test_data, train=False,disc=disc)

    def iteration(self, epoch, data_loader, train=True,disc=False):
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
        model_mode = "disc" if disc else "gen"



        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%s:%d" % (str_code, model_mode,epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        cumulative_loss = 0.0
        total_corect = 0
        total_mask_correct = 0
        g_total_correct = 0
        g_total_3_mask_correct = 0
        g_total_5_mask_correct = 0
        g_total_10_mask_correct = 0
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

            if disc or not train:
                self.electra_gen.eval()
                with torch.no_grad():
                    g_loss,g_scores = self.electra_gen(data['electra_input'],mask,data['electra_mask_label'])
            else:
                self.electra_gen.train()
                g_loss,g_scores = self.electra_gen(data['electra_input'],mask,data['electra_mask_label'])
            
            if disc:
                with torch.no_grad():
                    g_predictions = g_scores.max(2)[1]
                    disc_labels = (data["electra_label"] != g_predictions).long()
                    disc_labels = disc_labels.masked_fill(~data["mask_locations"],0)
                    disc_inputs = torch.where(data["mask_locations"],g_predictions,data["electra_input"])
                if not train:
                    self.electra_disc.eval()
                    with torch.no_grad():
                        d_loss,d_scores = self.electra_disc(disc_inputs,mask,disc_labels)
                else:
                    self.electra_disc.train()
                    d_loss,d_scores = self.electra_disc(disc_inputs,mask,disc_labels)   

                del g_predictions
                del disc_inputs 


            # 3. backward and optimization only in train
            if train and disc:
                    #self.optim_schedule.zero_grad()
                    self.optimd.zero_grad()
                    if self.hardware == "parallel":
                        d_loss.mean().backward()
                    else:
                        d_loss.backward()
                    #self.optim_schedule.step_and_update_lr()
                    self.optimd.step()
            elif train and not disc:
                    #self.optim_schedule.zero_grad()
                    self.optimg.zero_grad()
                    if self.hardware == "parallel":
                        g_loss.mean().backward()
                    else:
                        g_loss.backward()
                    #self.optim_schedule.step_and_update_lr()
                    self.optimg.step()

            #get generator accuracy for masked tokens
            g_predictions = g_scores.sort(2,descending=True)[1]
            g_mask_predictions = torch.masked_select(g_predictions,data["mask_locations"].unsqueeze(2).expand(g_predictions.shape[0],g_predictions.shape[1],g_predictions.shape[2])).reshape((data["mask_locations"].sum().item(),g_predictions.shape[2]))
            g_mask_token_labels = torch.masked_select(data["electra_mask_label"],data["mask_locations"])
            g_total_mask_correct += torch.sum(g_mask_predictions[:,0] == g_mask_token_labels).item()
            g_total_3_mask_correct += ELECTRATrainer.check_top_x_mask_predictions(g_mask_predictions,g_mask_token_labels,3)
            g_total_5_mask_correct += ELECTRATrainer.check_top_x_mask_predictions(g_mask_predictions,g_mask_token_labels,5)
            g_total_10_mask_correct += ELECTRATrainer.check_top_x_mask_predictions(g_mask_predictions,g_mask_token_labels,10)        
            total_mask += g_mask_token_labels.shape[0] 

            del g_mask_predictions
            del g_mask_token_labels

            #get generator accuracy for all tokens
            g_total_correct += torch.masked_select((g_predictions[:,:,0] == data["electra_label"]),mask.bool()).sum().item()

            del g_predictions
            del g_scores
            if disc:
                #get discriminator accuracy for all tokens
                d_predictions = torch.where(d_scores > 0.5,torch.tensor([1]).to(self.device),torch.tensor([0]).to(self.device))
                d_mask_predictions = torch.masked_select(d_predictions,data["mask_locations"])
                d_mask_token_labels = torch.masked_select(disc_labels,data["mask_locations"])
                d_total_mask_correct += torch.sum(d_mask_predictions == d_mask_token_labels).item()    

                del d_mask_predictions
                del d_mask_token_labels       
                
                #get discriminator accuracy for all tokens
                d_total_correct += torch.masked_select((d_predictions == disc_labels),mask.bool()).sum().item()
            total_element += mask.sum().item()


            log_loss = 0
            if self.hardware == "parallel":

                if disc:
                    cumulative_loss += d_loss.sum().item()
                    log_loss = d_loss.sum().item()
                    total_correct = d_total_correct
                    total_mask_correct = d_total_mask_correct
                else:
                    cumulative_loss += g_loss.sum().item()
                    log_loss = g_loss.sum().item()
                    total_correct = g_total_correct
                    total_mask_correct = g_total_mask_correct
                
            else: 
                if disc:
                    cumulative_loss += d_loss.item()
                    log_loss = d_loss.item()
                    total_correct = d_total_correct
                    total_mask_correct = d_total_mask_correct
                else:
                    cumulative_loss += g_loss.item()
                    log_loss = g_loss.item()
                    total_correct = g_total_correct
                    total_mask_correct = g_total_mask_correct
            if i % self.log_freq == 0 and total_mask > 0:
                #pdb.set_trace()
                data_iter.write("epoch: {}, iter: {}, avg loss: {},accuracy: {}/{}={:.2f}%, mask accuracy: {}/{}={:.2f}%, loss: {}".format(epoch,i,cumulative_loss/((i+1)*data_loader.batch_size),total_correct,total_element,total_correct/total_element*100,total_mask_correct,total_mask,total_mask_correct/total_mask*100,log_loss))

 
            del data
            del mask
            if disc:
                del d_scores
                del disc_labels
                del d_predictions


        print("EP{}_{}_{}, avg_loss={}, accuracy={:.2f}%".format(epoch,model_mode,str_code,cumulative_loss /(len(data_iter)*data_loader.batch_size),total_mask_correct/total_mask*100))
        if disc and self.d_log_file:
            f = open(self.d_log_file,"a")
            f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,str_code,cumulative_loss/(len(data_loader)*data_loader.batch_size),total_correct,total_element,total_correct/total_element*100,total_mask_correct,total_mask,total_mask_correct/total_mask*100))
        elif self.g_log_file:
            f = open(self.g_log_file,"a")
            f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,str_code,cumulative_loss/(len(data_loader)*data_loader.batch_size),total_correct,total_element,total_correct/total_element*100,total_mask_correct,g_total_3_mask_correct,g_total_5_mask_correct,g_total_10_mask_correct,total_mask,total_mask_correct/total_mask*100))

 
    @staticmethod
    def check_top_x_mask_predictions(mask_predictions,mask_labels,x):
        return torch.sum(mask_predictions[:,:x] == mask_labels.unsqueeze(1).expand((mask_labels.shape[0],x))).item()
        
    def save(self, epoch, file_path,disc=False):
        """
        Saving the current ELECTRA model on file_path

        :param epoch: current epoch number
        :param file_path: model output directory
        """
        output_file_path = file_path+"_epoch{}".format(epoch)
        if self.hardware == "parallel":
            #pdb.set_trace()
            if disc:
                self.electra_disc.module.discriminator.save_pretrained(output_file_path+"_disc")
                torch.save(self.electra_disc.module.embed_layer.state_dict(),output_file_path+"disc_embed")
            else:
                self.electra_gen.module.generator.save_pretrained(output_file_path+"_gen")
                torch.save(self.electra_gen.module.embed_layer.state_dict(),output_file_path+"gen_embed")
        else:
            if disc:
                self.electra_disc.discriminator.save_pretrained(output_file_path+"_disc")
                torch.save(self.electra_disc.embed_layer.state_dict(),output_file_path+"disc_embed")
            else:
                self.electra_gen.generator.save_pretrained(output_file_path+"_gen")
                torch.save(self.electra_gen.embed_layer.state_dict(),output_file_path+"gen_embed")
