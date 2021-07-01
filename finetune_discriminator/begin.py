import argparse
import pdb
from torch.utils.data import DataLoader
import numpy as np
import torch
from pretrain_hf import ELECTRATrainer
from dataset import ELECTRADataset,create_class_weights,create_weighted_sampler
from transformers import ElectraConfig,ElectraForSequenceClassification
from electra_discriminator import ElectraDiscriminator
from sklearn.model_selection import KFold

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--samples", required=True, type=str, help="microbiome samples")
    parser.add_argument("-tl", "--sample_labels",required=True,type=str,default=None, help="labels for samples")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with electra-vocab")
    parser.add_argument("-o", "--output_path", required=False, type=str, help="path to save models ex)output/")
    parser.add_argument("--test_samples",required=False,type=str,help="microbiome samples for test set, only provide if not trying to do cross validation. If providing this, --samples should provide the path to the training samples")
    parser.add_argument("--test_labels",required=False,type=str,help="labels for test set, only provide if not wanting to perform cross validation. If provided --sample_labels should provide the path to the train labels")
    parser.add_argument("--val_samples",required=False,type=str,help="microbiome samples for validation set, only provide if not trying to do cross validation. If providing this, --samples should provide the path to the training samples")
    parser.add_argument("--val_labels",required=False,type=str,help="labels for validation set, only provide if not wanting to perform cross validation. If provided --sample_labels should provide the path to the train labels")    

    parser.add_argument("-hs", "--hidden", type=int, default=100, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=5, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=10, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=1898, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--freeze_opt", type=int, default=0, help="parameter for choosing whether to freeze embeds or not, 0 means no freeze, 1 means all embeds are frozen, 2 means all embeds except cls are frozen")
    parser.add_argument("--freeze_encoders", type=int, default=0, help="parameter for choosing how many encoder layers to freeze")    


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

    parser.add_argument("--weighted_sampler", dest='class_imb_strat', action='store_true',help="use weighted sampler")
    parser.add_argument("--class_weights",dest='class_imb_strat',action='store_false',help="use class weights")
    parser.set_defaults(class_imb_strat=True)    

    parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--log_file", type=str,default=None,help="log file for performance metrics" )

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    parser.add_argument("--load_disc", type=str, default=None, help="path to saved state_dict of ELECTRA discriminator")
    parser.add_argument("--load_embed", type=str, default=None, help="path to saved state_dict of ELECTRA discriminator embedding layer")
    parser.add_argument("--num_labels", type = int, default = 2, help="number of labels for classification task")
    
    parser.add_argument("--resume_epoch", type=int, default=0, help="epoch to resume training at")    
    args = parser.parse_args()

    samples = np.load(args.samples)
    labels = np.load(args.sample_labels)


    def train_constructor(train_samples,test_samples,train_labels,test_labels,log_file,val_samples=None,val_labels=None):
            print("Loading Train Dataset")
            train_dataset = ELECTRADataset(train_samples, args.vocab_path,train_labels)

            print("Loading Test Dataset")
            test_dataset = ELECTRADataset(test_samples, args.vocab_path,test_labels)

            if val_samples is not None and val_labels is not None:
                val_dataset = ELECTRADataset(val_samples,args.vocab_path,val_labels)
                val_data_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers)                
            
            class_weights = None

            print("Creating Dataloader")


            if args.class_imb_strat:
                sampler = create_weighted_sampler(train_labels)
                train_data_loader = DataLoader(train_dataset,sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers)
            else:
                class_weights = create_class_weights(train_labels)
                train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True)

            train_orig_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=False)
            test_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)


            vocab_len = train_dataset.vocab_len()
        

            electra_config = ElectraConfig(vocab_size=vocab_len,embedding_size=args.hidden,hidden_size=args.hidden*2,num_hidden_layers=args.layers,num_attention_heads=args.attn_heads,intermediate_size=4*args.hidden,max_position_embeddings=args.seq_len,num_labels=args.num_labels)
            electra = ElectraDiscriminator(electra_config,torch.from_numpy(train_dataset.embeddings),args.load_disc,args.load_embed)
            print(electra)
            #pdb.set_trace()
            print("Creating Electra Trainer")
            if args.class_imb_strat:
                trainer = ELECTRATrainer(electra, vocab_len, train_dataloader=train_data_loader,train_orig_dataloader = train_orig_dataloader, test_dataloader=test_data_loader,val_dataloader=val_data_loader,
                                    lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                    with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,log_file=log_file,
                                    freeze_embed=args.freeze_opt,freeze_encoders=args.freeze_encoders,loss_func=args.loss_func,optim=args.optim,hidden_size=args.hidden*2)
            else:
                trainer = ELECTRATrainer(electra, vocab_len, train_dataloader=train_data_loader,train_orig_dataloader = train_orig_dataloader, test_dataloader=test_data_loader,val_dataloader=val_data_loader,
                                lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,log_file=log_file,
                                freeze_embed=args.freeze_opt,freeze_encoders=args.freeze_encoders,class_weights=torch.tensor(class_weights,dtype=torch.float),loss_func=args.loss_func,optim=args.optim,hidden_size=args.hidden*2)

            print("Training Start")
            for epoch in range(args.resume_epoch,args.epochs+args.resume_epoch):
                #pdb.set_trace()
                trainer.train(epoch,args.multi)
                #if epoch == 4 or epoch == 9 or epoch == 14 or epoch == 19:
                    #trainer.save(epoch, args.output_path)
                    #pdb.set_trace()

                trainer.train_orig_dist(epoch,args.multi)
                if val_samples is not None and val_labels is not None:
                    trainer.val(epoch,args.multi)
                trainer.test(epoch,args.multi)

    if args.test_samples is not None and args.test_labels is not None and args.val_samples is None and args.val_labels is None:
        test_samples = np.load(args.test_samples)
        test_labels = np.load(args.test_labels)
        #pdb.set_trace()
        log_file = args.log_file+".txt"
        train_constructor(samples,test_samples,labels,test_labels,log_file)

    elif args.test_samples is not None and args.test_labels is not None and args.val_samples is not None and args.val_labels is not None:
        test_samples = np.load(args.test_samples)
        test_labels = np.load(args.test_labels)
        val_samples = np.load(args.val_samples)
        val_labels = np.load(args.val_labels)
        #pdb.set_trace()
        log_file = args.log_file+".txt"
        train_constructor(samples,test_samples,labels,test_labels,log_file,val_samples,val_labels)


    else:
        split_count = 1
        kf = KFold(n_splits=5,shuffle=True,random_state=42)
        for train_index,test_index in kf.split(samples):
            log_file = args.log_file+"_valset"+str(split_count)+".txt"
            train_samples = samples[train_index]
            train_labels = labels[train_index]
            test_samples = samples[test_index]
            test_labels = labels[test_index]

            train_constructor(train_samples,test_samples,train_labels,test_labels,log_file)
            split_count += 1

if __name__ == "__main__":
    train()