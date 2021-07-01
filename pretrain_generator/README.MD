# Finetune Discriminator

This folder provides code for pretraining a generator on microbiome data

## Environment Setup
\>=Python 3.5

Required Python packages
* Pytorch
* Huggingface Transformers

## Running Code

```begin.py``` provides a command line interface that allows specification of paths to training and testing data and labels, path to output folder for saving, path to output folder for logging metrics, and model parameters such as learning rate, batch size, number of encoder layers, number of attention heads, optimizer, and loss function. No set directory structure for data files is required as all paths are specified in the command line arguments

```python begin.py --train_dataset ~/guille/train_encodings_512.npy --test_dataset ~/guille/test_encodings_512.npy --vocab_path ~/guille/vocab_embeddings.npy --output_path ~/models/gensgdlr1e2/gen --batch_size 32 --layers 10 -e 200 --attn_heads 10 --seq_len 513 --cuda --log_file ~/models/gensgdlr1e2/gen.txt --resume_epoch 200 --load_gen ~/models/gensgdlr1e2/gen_epoch199_gen/pytorch_model.bin --load_g_embed ~/models/gensgdlr1e2/gen_epoch199_gen_embed ```

### Command Line Interface Legend

The following commands are required:
* ```--train_dataset, -c```:  Path to numpy array of dimension (# samples, Sequence Length (e.g. 512), 2) with the 2 representing (Embedding id, Frequency). Represents training examples when validation and test set are specified, otherwise represents set of all samples to be broken up into cross validation.
* ```--test_dataset, -t```: Path to numpy array for test set, same dimension constraints as for --train_dataset
*  ``` --vocab_path, -v```: Path to numpy array containing microbe embeddings, expected shape is (# of embeddings, embedding_dimension)
 * ```--output_path,-o```:Path to prefix for saving models.  It is a prefix so that the code can save models at various epochs. As an example if you wanted your models saved with the title "firstmodel" in directory "~/output", the correct argument would be --output_path ~/output/firstmodel. The code will generate the necessary extensions.
 
 The following commands are optional:

* ```--log_file```: Path to log file for performance metrics logging e.g. --log_file ~/output/log_firstmodel.txt
* ```--resume_epoch```: If resuming training, specify which epoch training is resuming at.
 * ```--load_gen```: path to saved state_dict of Masked LM model. Used for resuming training.
 * ```--load_g_embed```: path to saved state dict for generator embeddings. Used for resuming training.
 * ```--hidden, -hs```: Hidden size of transformer model, default value of 100
 * ```--layers,-l```: Number transformer encoder layers, default value of 5
 * ```--attn_heads,-a```: Number of attention heads in encoder layer, default value of 10
 * ```--seq_len, -s```: Maximum sequence length for sample, default value of 1898. Should be set to 1 more than maximum sequence length in your samples to accommodate cls token, e.g. for samples with max length of 512, this should be set to 513
 * ```--batch_size,-b:``` batch size for training, default 32
 * ```--epochs,-e:```number of epochs, default 10 
  * ```--cuda```: Train with cuda if available
 * ```--no_cuda```: Train on cpu. Default setting is to train on cpu
 * ```--lr```: Specify learning rate, default 0.01
  * ```--log_freq```: specify n for printing loss to command line every n iterations, default 100
 * ```--cuda_devices```: If using multiple cuda devices, specify which devices to use, otherwise code will default to using all available gpus. Example usage --cuda devices 0 1 3

## Other important Notes
### Optimization
By default  this uses SGD optimization. There is commented out code for using adam but it is outdated and doesn't use the proper learning rate schedule, please refer to the implementation of Adam in the finetune_discriminator folder as a reference for adding in Adam. The command line options for specifying adams parameters are present but a switch for choosing between sgd and adam is not implemented.

### Freezing Embeddings
All embeddings are trained. Freezing of embeddings has not been implemented. Please refer to the finetune_discriminators folder to see how freezing can be handled if you wish to add it here.



