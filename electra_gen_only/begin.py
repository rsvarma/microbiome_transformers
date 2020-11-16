import argparse
import pdb
from torch.utils.data import DataLoader

import torch
from pretrain_hf import ELECTRATrainer
from dataset import ELECTRADataset
from transformers import ElectraConfig
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

    parser.add_argument("-b", "--batch_size", type=int, default=3, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--cuda", dest='with_cuda', action='store_true',help="train with CUDA")
    parser.add_argument("--no_cuda",dest='with_cuda',action='store_false',help="train on CPU")
    parser.set_defaults(with_cuda=False)

    parser.add_argument("--loss_weight", type=float,default = 50.0, help="weight assigned to discriminator loss")

    parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--log_file", type=str,default=None,help="log file for performance metrics" )

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("--load_gen", type=str, default=None, help="path to saved state_dict of Masked LM model")
    parser.add_argument("--load_g_embed", type=str, default=None, help="path to saved state dict for generator embeddings")
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
  

    electra_config = ElectraConfig(vocab_size=vocab_len,embedding_size=args.hidden,hidden_size=2*args.hidden,num_hidden_layers=args.layers,num_attention_heads=args.attn_heads,intermediate_size=4*args.hidden,max_position_embeddings=args.seq_len,)
    electra = ElectraGenerator(electra_config,torch.from_numpy(train_dataset.embeddings),args.load_gen,args.load_g_embed)
    print("Creating Electra Trainer")

    append = True if args.resume_epoch > 0 else False
    trainer = ELECTRATrainer(electra, vocab_len, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,log_file=args.log_file,
                          append=append)

    print("Training Start")
    for epoch in range(args.resume_epoch,args.resume_epoch+args.epochs):
        trainer.train(epoch)
        if epoch == 0 or (epoch+1)%10 == 0:
            trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)


if __name__ == "__main__":
    train()