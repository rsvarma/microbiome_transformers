# Finetune Discriminator

This folder provides code for finetuning a discriminator on a binary classification task

## Environment Setup
\>=Python 3.5

Required Python packages
* Pytorch
* Scikit-learn
* Huggingface Transformers

## Running Code

```begin.py``` provides a command line interface that allows specification of paths to training, validation, and testing data and labels, path to output folder for saving, path to output folder for logging metrics, and model parameters such as learning rate, batch size, number of encoder layers, number of attention heads, optimizer, and loss function. No set directory structure for data files is required as all paths are specified in the command line arguments

### Example for finetuning on specified training, validation, and testing sets

```
python begin.py --samples ~/data/train_otu.npy --sample_labels ~/data/train_labels.npy --val_samples ~/data/val_otu.npy --val_labels ~/data/val_labels.npy --test_samples ~/data/test_otu.npy --test_labels ~/data/test_labels.npy --vocab_path ~/data/vocab_embeddings.npy --output_path ~/output/modelname --batch_size 32 --layers 5 --epochs 50 --attn_heads 5 --cuda --log_file ~/models/test --freeze_opt 2 --weighted_sampler --seq_len 513 --num_labels 2 --load_disc ~/models/discsgdlr1e2/frozen_embeds/5head5layer_epoch120_disc/pytorch_model.bin --mse --sgd --lr 0.01
```
### Example for finetuning with 5-fold cross validation

In this case, only one numpy array of samples and labels are provided as the code will divide the samples up into folds. Note that if cross validation is desired, a separate validation or test set using the --validation_samples or --test_samples commands should not be used. 

```
python begin.py --samples ~/guille/FRUIT_FREQUENCY_otu_512.npy --sample_labels ~/guille/FRUIT_FREQUENCY_binary34_labels.npy --vocab_embeddings ~/guille/vocab_embeddings.npy --output_path ~/models/fineadamlr1e2/partially_frozen/240epgen5head5layer/fruit5l5h/fruit5l5hMSEbin34 --batch_size 32 --layers 5 --epochs 50 --attn_heads 5 --cuda --log_file ~/models/fineadamlr1e2/partially_frozen/240epgen5head5layer/fruit5l5h/fruit5l5hMSEbin34 --freeze_opt 2 --weighted_sampler -s 513 --num_labels 2 --load_disc ~/models/discsgdlr1e2/frozen_embeds/240epgen/5head5layer_epoch120_disc/pytorch_model.bin --mse --adam --lr 0.01

```

### Command Line Interface Legend

The following commands are required:

* ``` --samples, -t```: Path to numpy array of dimension (# samples, Sequence Length (e.g. 512), 2) with the 2 representing (Embedding id, Frequency). Represents training examples when validation and test set are specified, otherwise represents set of all samples to be broken up into cross validation.
* ```--sample_labels, -tl```: Path to numpy array containing labels for classification task (e.g. numpy array containing 1s and 0s corresponding to IBD status) for samples specified with --samples
*  ``` --vocab_path, -v```: Path to numpy array containing microbe embeddings, expected shape is (# of embeddings, embedding_dimension)

The following commands are optional:
 * ```--log_file```: Specify path to logging file for metrics. Note that this should be a path to a prefix, e.g if you wanted to save to ~/output/firstmodel.txt, the correct argument would be --log_file ~/output/firstmodel. This is because if cross validation is being done the code will generate log files based on the prefix that also provide information about which fold the file pertains to, e.g., ~/output/firstmodel_valset1.txt. Even if cross validation is not being done, the code will handle appending ".txt" onto log files.
 * ```--output_path,-o```:Path to prefix for saving models. Just like with --log_file this is a prefix, so if you want to save your model as firstmodel in directory output, the correct argument would be --output_path ~/output/firstmodel. The code will generate the necessary extensions. Also note that currently the code does not save the models, if you want the models to be saved please uncomment and adjust the if statement 
	 ```
	 #if epoch == 4 or epoch == 9 or epoch == 14 or epoch == 19:
		#trainer.save(epoch, args.output_path)
	 ```
	 at the bottom of the  ```train_constructor``` function in ```begin.py```
 * ```--load_disc```: Path to saved state_dict of ELECTRA discriminator. If this is provided the model will be initialized to the specified discriminator as opposed to being trained from scratch
 * ```--load_embed``: Path to saved state of ElECTRA discriminators embedding layer. If provided, model's embedding layer will be initialized based on the provided layer.
 * ```--resume_epoch```: If resuming training, specify which epoch training is resuming at.
* ```--val_samples ```: Path to numpy array for validation set, same dimension constraints as for --samples
 * ```--test_samples ```: Path to numpy array for test set, same dimension constraints as for --samples
 * ```--val_labels```: Path to numpy array for validation set labels
 * ```--test_labels```: Path to numpy array for test set labels
 * ```--hidden, -hs```: Hidden size of transformer model, default value of 100
 * ```--layers,-l```: Number transformer encoder layers, default value of 5
 * ```--attn_heads,-a```: Number of attention heads in encoder layer, default value of 10
 * ```--seq_len, -s```: Maximum sequence length for sample, default value of 1898. Should be set to 1 more than maximum sequence length in your samples to accommodate cls token, e.g. for samples with max length of 512, this should be set to 513
 * ```--batch_size,-b:``` batch size for training, default 32
 * ```--epochs,-e:```number of epochs, default 10
 * ```--weighted_sampler```: Specify to use weighted sampling as strategy for dealing with class imblance in training data. This is the default behavior
 * ```--class_weights```: Specify to use class weighting in the loss function as the strategy for dealing with class imbalance. This option hasnt been tested in a while so it is not recommended.
 * ```--freeze_opt```: Int specifying whether embedding layer should be frozen or not. 0 means embedding layer is not frozen, 1 means embedding layer is frozen, 2 means all embeddings except cls token are frozen. Default value 0
 * ```--freeze_encoders```: Number of encoder layers to freeze. Default value 0
 * ```--cuda```: Train with cuda if available
 * ```--no_cuda```: Train on cpu. Default setting is to train on cpu
 * ```--ce```: Use cross entropy loss
 * ```--mse```: Use mse loss
 * ```--adam```: Use adam optimizer
 * ```--sgd```:Use sgd optimizer. Default Optimizer
 * ```--lr```: Specify learning rate, default 0.01
 * ```--adam_weight_decay```": Adam weight decay parameter, default 0.01
 * ```--adam_beta1```: Adam first beta value, default 0.9
 * ```--adam_beta2```: Adam second beta value, default 0.999
 * ```--log_freq```: specify n for printing loss to command line every n iterations, default 100
 * ```--cuda_devices```: If using multiple cuda devices, specify which devices to use, otherwise code will default to using all available gpus. Example usage --cuda devices 0 1 3
 * ```--num_labels```: Specifies number of unique labels classification task, default 2.
 * ```--multi```: Include if number of labels is more than 2 



