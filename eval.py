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

def parse_args():
    parser = argparse.ArgumentParser(description="NLI Evaluation task")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--encoder_model", type=str, default="mean", help="Choices would be, uni-lstm, bi-lstm, bi-max-lstm, mean")
    parser.add_argument("--epochs", type=int, default=20) # It converges based on the LR so best to let it run for a while with a higher value
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout_encoder", type=float, default=0.0)
    parser.add_argument("--dropout_classifier", type=float, default=0.0)
    parser.add_argument("--lstm_dim", type=int, default=512)
    parser.add_argument("--classifier_dim", type=int, default=512)
    parser.add_argument("--class_count", type=int, default=3)
    parser.add_argument("--sgd_lr", type=float, default=0.1)
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.99)
    # Paths
    parser.add_argument("--outputdir", type=str, default='modelsaves/')
    parser.add_argument("--checkpoint_path", type=str, default='modelsaves/mean_model.pth')
    parser.add_argument("--outputmodelname", type=str, default='model_output')
    parser.add_argument("--glovepath", type=str, default='glove.840B.300d.txt')
    parser.add_argument("--sentevalpath", type=str, default='../SentEval')
