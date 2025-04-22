import argparse
from models import SNLIModel
import torch
import numpy as np
import os
import sys
import numpy as np
import logging
import sklearn
from utils import create_batch, preprocess_string
import senteval
from train import parse_args

def prepare(params, samples):
    # _, params.word2id = data.create_dictionary(samples)
    # # load glove/word2vec format 
    # params.word_vec = data.get_wordvec(params.glovepath, params.word2id)
    # # dimensionality of glove embeddings
    # params.wvec_dim = 300
    print("Next task!")
    return

def batcher(params, batch):
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sentence if sentence != [] else ['.'] for sentence in batch]
    # print(batch)
    # This works because the batch seemed to be already tokenized when i checked it
    processed_batch = create_batch(batch, params.model.glove_dict)
    embeddings = params.model.classifier.encode(processed_batch)

    # [batch size, embedding dimensionality]
    return embeddings.detach().cpu()

# Example command!
# python eval.py --encoder_model uni-lstm --checkpoint_path modelsaves/uni_lstm_model.pth
# Make sure that the encoder model and checkpoint match otherwise u get errors!

if __name__ == "__main__":
    params = parse_args()
    # Load the model through the checkpoint (just give the checkpoint argument, and it will load it)
    model = SNLIModel(params)
    params_senteval = {'task_path': params.sentevalpath+"/data", 'usepytorch': False, 'kfold': 2, 'model': model}
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['MR', 'CR']
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ']
    results = se.eval(transfer_tasks)
    print(results)
    # Compute the macro and micro accuracy
    accuracies = [v['acc'] for v in results.values()]
    ntests = [v['ntest'] for v in results.values()]
    macro_accuracy = np.mean(accuracies)

    total_samples = sum(ntests)
    weighted_sum = sum(accuracy * n for accuracy, n in zip(accuracies, ntests))
    micro_accuracy = weighted_sum / total_samples

    print(f"Macro Accuracy: {macro_accuracy:.2f}")
    print(f"Micro Accuracy: {micro_accuracy:.2f}")
