# Dependencies and package installation guide
The code is written using Python 3.13.2 and requires you to download the following datasets to make use of it all.

Glove embeddings Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors): https://nlp.stanford.edu/projects/glove/

SentEval for eval.py (follow the guidelines in their github): https://github.com/facebookresearch/SentEval

Then to install all dependencies make use of the environment.yml file and the following command conda env create -f environment.yml and conda activate ACTS

Make sure after activating the conda environment to also additionally pip install SentEval into the environment after cloning their github 
Do this by running the command: pip install . inside of the SentEval root folder

# Structure of the code and file explanation
- models.py, this file contains all 4 encoder models and a simple MLP based classifier model additionally it contains a SNLIModel class that is used to train and evaluate
- train.py, uses the SNLIModel class from models.py you can give the file various arguments when running it to alter the training parameters (explained below)
- utils.py, contains helper functions used by models.py most of these functions make use of the glove embeddings. So make sure to have those installed
- eval.py, file to evaluate a model checkpoint using the SentEval dataset
- demo.ipynb a demonstration on how to use the checkpoints to predict a label based on a NLI task

# How to train
You can use the train.py file to train 4 different encoders.
The train.py file accepts the following arguments
## Command-Line Arguments

The following command-line arguments are available for configuring the model training process:

| Argument              | Type    | Default Value           | Description                                                                                                                                   |
|-----------------------|---------|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `--seed`              | `int`   | `1234`                  | Random seed for reproducibility.                                                                                                               |
| `--encoder_model`     | `str`   | `"uni-lstm"`            | Encoder model type. Options: `uni-lstm`, `bi-lstm`, `bi-max-lstm`, `mean`.                                                                   |
| `--epochs`            | `int`   | `20`                    | Number of training epochs. It is recommended to let it run for a while with a higher value depending on learning rate.                          |
| `--batch_size`        | `int`   | `64`                    | Size of each batch during training.                                                                                                            |
| `--dropout_encoder`   | `float` | `0.0`                   | Dropout rate for the encoder.                                                                                                                 |
| `--dropout_classifier`| `float` | `0.0`                   | Dropout rate for the classifier.                                                                                                              |
| `--lstm_dim`          | `int`   | `512`                   | Dimension of the LSTM layer.                                                                                                                  |
| `--classifier_dim`    | `int`   | `512`                   | Dimension of the classifier layer.                                                                                                            |
| `--class_count`       | `int`   | `3`                     | Number of output classes for classification.                                                                                                  |
| `--sgd_lr`            | `float` | `0.1`                   | Learning rate for Stochastic Gradient Descent (SGD).                                                                                          |
| `--sgd_momentum`      | `float` | `0.9`                   | Momentum value for SGD.                                                                                                                       |
| `--decay`             | `float` | `0.99`                  | Decay rate for the learning rate.                                                                                                             |
| **Paths**             |         |                         |                                                                                                                                               |
| `--outputdir`         | `str`   | `'modelsaves/'`         | Directory to save model checkpoints and outputs.                                                                                             |
| `--checkpoint_path`   | `str`   | `'none'`                | Path to a saved checkpoint model (if resuming training).                                                                                      |
| `--outputmodelname`   | `str`   | `'model_output'`        | Name of the output model file.                                                                                                                |
| `--glovepath`         | `str`   | `'glove.840B.300d.txt'` | Path to the GloVe word embeddings file (use a pre-trained GloVe file).                                                                         |

### Example Usage
python train.py --epochs 50 --encoder_model bi-lstm --outputmodelname bi_lstm_model

# How to evaluate with SentEval
You can evaluate model checkpoints using the eval.py file
It evaluates on the following transfer tasks
'MR', 'CR', 'MPQA', 'SUBJ'
### Example Usage
python eval.py --encoder_model mean --checkpoint_path modelsaves/mean_model.pth

You need to make sure that the encoder_model and checkpoint match. If they do not, the file will fail to evaluate.


  

