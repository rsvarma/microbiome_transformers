# Pretrain Discriminator

This folder provides code for pretraining a discriminator on specified pretrained generators. More specifically, it trains the discriminator on each specified generator for a set number of epochs. Since discriminators and generators may have different architectures (number of layers, attention heads etc) it is important to specify both.

## Environment Setup
\>=Python 3.5

Required Python packages
* Pytorch
* Huggingface Transformers

## Running Code

```begin.py``` provides a command line interface that allows specification of paths to training and testing data and labels, path to output folder for saving, path to output folder for logging metrics, paths to generators to load, and model parameters such as learning rate, batch size, number of encoder layers, number of attention heads, optimizer, and loss function. No set directory structure for data files is required as all paths are specified in the command line arguments

```
python begin.py -c ~/guille/train_encodings_512.npy -t ~/guille/test_encodings_512.npy -v ~/guille/vocab_embeddings.npy -o ~/models/discsgdlr1e2/frozen_embeds/240epgen/5head5layer -b 32 -l 5 -a 5 -e 15 -s 513 -gl 10 -ga 10 -gs 513 --cuda --d_log_file ~/models/discsgdlr1e2/frozen_embeds/240epgen/5head5layer.txt --load_gen ~/models/gensgdlr1e2/gen_epoch29_gen/pytorch_model.bin ~/models/gensgdlr1e2/gen_epoch59_gen/pytorch_model.bin ~/models/gensgdlr1e2/gen_epoch89_gen/pytorch_model.bin ~/models/gensgdlr1e2/gen_epoch119_gen/pytorch_model.bin ~/models/gensgdlr1e2/gen_epoch149_gen/pytorch_model.bin ~/models/gensgdlr1e2/gen_epoch179_gen/pytorch_model.bin ~/models/gensgdlr1e2/gen_epoch209_gen/pytorch_model.bin ~/models/gensgdlr1e2/gen_epoch239_gen/pytorch_model.bin --load_g_embed ~/models/gensgdlr1e2/gen_epoch29_gen_embed ~/models/gensgdlr1e2/gen_epoch59_gen_embed ~/models/gensgdlr1e2/gen_epoch89_gen_embed ~/models/gensgdlr1e2/gen_epoch119_gen_embed ~/models/gensgdlr1e2/gen_epoch149_gen_embed ~/models/gensgdlr1e2/gen_epoch179_gen_embed ~/models/gensgdlr1e2/gen_epoch209_gen_embed ~/models/gensgdlr1e2/gen_epoch239_gen_embed --freeze
```


### Command Line Interface Legend

The following commands are required:
* ```--train_dataset, -c```:  Path to numpy array of dimension (# samples, Sequence Length (e.g. 512), 2) with the 2 representing (Embedding id, Frequency). Represents training examples when validation and test set are specified, otherwise represents set of all samples to be broken up into cross validation.
* ```--test_dataset, -t```: Path to numpy array for test set, same dimension constraints as for --train_dataset
*  ``` --vocab_path, -v```: Path to numpy array containing microbe embeddings, expected shape is (# of embeddings, embedding_dimension)
 * ```--output_path,-o```:Path to prefix for saving models.  It is a prefix so that the code can save models at various epochs. As an example if you wanted your models saved with the title "firstmodel" in directory "~/output", the correct argument would be --output_path ~/output/firstmodel. The code will generate the necessary extensions.
 * ```--load_gen```: paths to saved state_dicts of Masked LM model. 
 * ```--load_g_embed```: paths to saved state dicts for generator embeddings. 
 
 The following commands are optional:

* ```--d_log_file```: Path to log file for performance metrics logging e.g. --d_log_file ~/output/log_firstmodel.txt
* ```--resume_epoch```: If resuming training, specify which epoch training is resuming at.
 * ```--hidden, -hs```: Hidden size of discriminator model, default value of 100
 * ```--layers,-l```: Number transformer encoder layers in discriminator, default value of 5
 * ```--attn_heads,-a```: Number of attention heads in encoder layers of discriminator, default value of 10
 * ```--seq_len, -s```: Maximum sequence length for sample discriminator will accept, default value of 1898. Should be set to 1 more than maximum sequence length in your samples to accommodate cls token, e.g. for samples with max length of 512, this should be set to 513
  * ```--gen_hidden, -ghs```: Hidden size of generator model, default value of 100
 * ```--gen_layers,-gl```: Number transformer encoder layers in generator, default value of 10
 * ```--gen_attn_heads,-ga```: Number of attention heads in encoder layers of generator, default value of 10
 * ```--gen_seq_len, -gs```: Maximum sequence length for sample generator will accept, default value of 1898. Should be set to 1 more than maximum sequence length in your samples to accommodate cls token, e.g. for samples with max length of 512, this should be set to 513
 * ```--batch_size,-b:``` batch size for training, default 32
 * ```--epochs,-e:```number of epochs to train on each generator, default 10 
  * ```--cuda```: Train with cuda if available
 * ```--no_cuda```: Train on cpu. Default setting is to train on cpu
 * ```--lr```: Specify learning rate, default 0.01
  * ```--log_freq```: specify n for printing loss to command line every n iterations, default 100
 * ```--cuda_devices```: If using multiple cuda devices, specify which devices to use, otherwise code will default to using all available gpus. Example usage --cuda devices 0 1 3
 * ```--freeze```: Specifies to freeze embedding layer when training discriminator
 * ```--no_freeze```: Specifies to train embedding layer. This is default behavior

## Other important Notes
### Optimization
By default  this uses SGD optimization. There is commented out code for using adam but it is outdated and doesn't use the proper learning rate schedule, please refer to the implementation of Adam in the finetune_discriminator folder as a reference for adding in Adam. The command line options for specifying adams parameters are present but a switch for choosing between sgd and adam is not implemented.

### Generator Architecture
It should be noted that only one set of parameters is specified for the generator architecture. Thus the code assumes that all generators specified have the same architecture (stemming from the codes assumption that the specified generators are the same model simply varying numbers of epochs into training).





