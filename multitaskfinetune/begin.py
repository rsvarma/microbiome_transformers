import os
import argparse
import pdb
from torch.utils.data import DataLoader
import numpy as np
import torch
from pretrain_hf import ELECTRATrainer
from dataset import ELECTRADataset,create_weighted_sampler
from transformers import ElectraConfig,ElectraForSequenceClassification
from electra_discriminator import ElectraDiscriminator
from sklearn.model_selection import KFold

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--samples", required=True, type=str, help="microbiome samples")
    parser.add_argument("-tl", "--sample_labels",required=True,type=str,default=None, help="labels for samples")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with electra-vocab")
    parser.add_argument("-o", "--output_path", required=False, type=str,default=None, help="ex)output/")

    parser.add_argument("-hs", "--hidden", type=int, default=100, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=10, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=1898, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=3, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--freeze_opt", type=int, default=0, help="parameter for choosing whether to freeze embeds or not, 0 means no freeze, 1 means all embeds are frozen, 2 means all embeds except cls are frozen")

    parser.add_argument("--cuda", dest='with_cuda', action='store_true',help="train with CUDA")
    parser.add_argument("--no_cuda",dest='with_cuda',action='store_false',help="train on CPU")
    parser.set_defaults(with_cuda=False)

    parser.add_argument("--multi", dest='multi', action='store_true',help="training on multiclass problem")
    parser.set_defaults(multi=False)


    parser.add_argument("--ce",dest='loss_func',action='store_const',const='ce',help="train with cross entropy loss")
    parser.add_argument("--mse",dest='loss_func',action='store_const',const='mse',help="train with mean square error loss loss")
    parser.set_defaults(loss_func='mse')
        
    parser.add_argument("--adam",dest='optim',action='store_const',const='adam',help="train with adam optimizer")
    parser.add_argument("--sgd",dest='optim',action='store_const',const='sgd',help="train with sgd")
    parser.set_defaults(optim='sgd')


    parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--log_dir", type=str,default=None,help="directory for logging performance metrics" )
    parser.add_argument("--task_names", type=str,nargs='+', default=None, help="names of tasks being trained on, in same order as in given labels file")

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("--load_disc", type=str, default=None, help="path to saved state_dict of ELECTRA discriminator")
    parser.add_argument("--load_embed", type=str, default=None, help="path to saved state_dict of ELECTRA discriminator embedding layer")
    parser.add_argument("--num_labels", type = int, default = 2, help="number of labels for classification task")
    
    parser.add_argument("--resume_epoch", type=int, default=0, help="epoch to resume training at")    
    args = parser.parse_args()


    samples = np.load(args.samples)
    labels = np.load(args.sample_labels)

    split_count = 1
    kf = KFold(n_splits=5,shuffle=True,random_state=42)
    for train_index,test_index in kf.split(samples):
        log_files = []
        for name in args.task_names:
            file_path = os.path.join(args.log_dir,name)
            log_files.append(file_path+"_valset"+str(split_count)+".txt")
        overall_log_file = os.path.join(args.log_dir,"log")+"_valset"+str(split_count)+".txt"
        train_samples = samples[train_index]
        train_labels = labels[train_index]
        test_samples = samples[test_index]
        test_labels = labels[test_index]

        print("Loading Train Dataset")
        train_orig_dataset = ELECTRADataset(train_samples, args.vocab_path,train_labels)

        print("Loading Test Dataset")
        test_dataset = ELECTRADataset(test_samples, args.vocab_path,test_labels)
        
        class_weights = None

        print("Creating Dataloader")


        #pdb.set_trace()
        train_data_loaders = []
        for i in range(len(args.task_names)):
            sampler = create_weighted_sampler(train_labels,i)
            task_dataset = ELECTRADataset(train_samples, args.vocab_path,train_labels,i)
            train_data_loaders.append(DataLoader(task_dataset,sampler=sampler, batch_size=args.batch_size//len(args.task_names), num_workers=args.num_workers))


        train_orig_dataloader = DataLoader(train_orig_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=False)
        test_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)

        vocab_len = train_orig_dataset.vocab_len()
    

        electra_config = ElectraConfig(vocab_size=vocab_len,embedding_size=args.hidden,hidden_size=args.hidden*2,num_hidden_layers=args.layers,num_attention_heads=args.attn_heads,intermediate_size=4*args.hidden,max_position_embeddings=args.seq_len,num_labels=args.num_labels)
        electra = ElectraDiscriminator(electra_config,len(args.task_names),torch.from_numpy(train_orig_dataset.embeddings),args.load_disc,args.load_embed)
        print(electra)
        print("Creating Electra Trainer")
        trainer = ELECTRATrainer(electra, vocab_len, train_dataloaders=train_data_loaders,train_orig_dataloader = train_orig_dataloader,task_log_files=log_files,log_file=overall_log_file, test_dataloader=test_data_loader,
                            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                            freeze_embed=args.freeze_opt,loss_func=args.loss_func,optim=args.optim,hidden_size=args.hidden*2)

        print("Training Start")
        for epoch in range(args.resume_epoch,args.epochs):
            #pdb.set_trace()
            trainer.train(epoch,args.multi)
            #if epoch == 4 or epoch == 9 or epoch == 14 or epoch == 19:
                #trainer.save(epoch, args.output_path)
                #pdb.set_trace()

            trainer.train_orig_dist(epoch,args.multi)
            trainer.test(epoch,args.multi)
        split_count += 1

if __name__ == "__main__":
    train()