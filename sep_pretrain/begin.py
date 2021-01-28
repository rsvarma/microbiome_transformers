import argparse
import pdb
from torch.utils.data import DataLoader

import torch
from pretrain_hf import ELECTRATrainer
from dataset import ELECTRADataset
from transformers import ElectraConfig
from electra_pretrain_model import ElectraDiscriminator
from electra_pretrain_model import ElectraGenerator


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train electra")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with electra-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/")

    parser.add_argument("-hs", "--hidden", type=int, default=100, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=10, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=1898, help="maximum sequence len")

    parser.add_argument("-ghs", "--gen_hidden", type=int, default=100, help="hidden size of gen transformer model")
    parser.add_argument("-gl", "--gen_layers", type=int, default=10, help="number of gen layers")
    parser.add_argument("-ga", "--gen_attn_heads", type=int, default=10, help="number of gen attention heads")
    parser.add_argument("-gs", "--gen_seq_len", type=int, default=1898, help="maximum gen sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=3, help="number of batch_size")
    parser.add_argument("-e","--epochs", type=int, default=10, help="number of epochs to train discriminator for each generator")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("--freeze", dest='freeze_embed', action='store_true',help="freeze discriminator embedding layer to GloVE embeddings")
    parser.add_argument("--no_freeze",dest='freeze_embed',action='store_false',help="train discriminator embedding layer after initializing to GloVE embeddings")
    parser.set_defaults(freeze_embed=False)

    parser.add_argument("--cuda", dest='with_cuda', action='store_true',help="train with CUDA")
    parser.add_argument("--no_cuda",dest='with_cuda',action='store_false',help="train on CPU")
    parser.set_defaults(with_cuda=False)


    parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--d_log_file", type=str,default=None,help="log file for discriminator performance metrics" )

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("--load_gen", type=str,nargs='+', default=None, help="path to saved state_dicts of Masked LM model")
    parser.add_argument("--load_g_embed", type=str,nargs='+', default=None, help="path to saved state_dicts for generator embeddings")
    parser.add_argument("--resume_epoch", type=int, default=0, help="epoch to resume training at")

    args = parser.parse_args()


    print("Loading Train Dataset", args.train_dataset)
    train_dataset = ELECTRADataset(args.train_dataset, args.vocab_path)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = ELECTRADataset(args.test_dataset, args.vocab_path) if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers,drop_last=True) \
        if test_dataset is not None else None

    vocab_len = train_dataset.vocab_len()
  

    electra_config = ElectraConfig(vocab_size=vocab_len,embedding_size=args.hidden,hidden_size=2*args.hidden,num_hidden_layers=args.layers,num_attention_heads=args.attn_heads,intermediate_size=4*args.hidden,max_position_embeddings=args.seq_len)
    electra_gen_config = ElectraConfig(vocab_size=vocab_len,embedding_size=args.gen_hidden,hidden_size=2*args.gen_hidden,num_hidden_layers=args.gen_layers,num_attention_heads=args.gen_attn_heads,intermediate_size=4*args.gen_hidden,max_position_embeddings=args.gen_seq_len)    
    electra_gen = ElectraGenerator(electra_gen_config,torch.from_numpy(train_dataset.embeddings),args.load_gen[0],args.load_g_embed[0])
    electra_disc = ElectraDiscriminator(electra_config,torch.from_numpy(train_dataset.embeddings))
    print("Creating Electra Trainer")
    append = True if args.resume_epoch > 0 else False
    trainer = ELECTRATrainer(electra_gen,electra_disc, vocab_len, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,d_log_file = args.d_log_file,
                          append=append,freeze_embed=args.freeze_embed)



    for i in range(len(args.load_gen)):
        new_gen = ElectraGenerator(electra_gen_config,torch.from_numpy(train_dataset.embeddings),args.load_gen[i],args.load_g_embed[i])
        trainer.swap_gen(new_gen,args.with_cuda,args.cuda_devices)
        trainer.save(0,args.output_path)
        for j in range(args.epochs):
            epoch_num = i*10+(j+1)
            trainer.train(epoch_num)
            trainer.test(epoch_num)
        trainer.save((i+1)*args.epochs,args.output_path)

if __name__ == "__main__":
    train()