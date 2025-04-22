import numpy as np
from datasets import load_dataset
import nltk
from datasets import DatasetDict
import torch

def preprocess_string(string):
    result = string.lower()
    result = nltk.word_tokenize(result) #i am using punkt_tab here!
    return result

def preprocess_dataset_example(example):
    example['premise'] = preprocess_string(example['premise'])
    example['hypothesis'] = preprocess_string(example['hypothesis'])
    return example

def filter_valid_labels(example):
    return example['label'] != -1

# This assumes to get the entire dataset Dict that snli gives you (all 3 sets)
def preprocess_data(data):
    preprocessed_data = {}
    for split in data.keys():
       processed_split  = data[split].map(preprocess_dataset_example)
       filtered_split = processed_split.filter(filter_valid_labels)
       preprocessed_data[split] = filtered_split
    
    return DatasetDict(preprocessed_data)

def build_vocab(dataset):
    vocab = set()
    for split in dataset:
        for example in dataset[split]:
            vocab.update(example["premise"])
            vocab.update(example["hypothesis"])
    return vocab

def create_embedding_dict(unique_vocab, path="glove.840B.300d.txt"):
    embeddings_dict = {}
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            try:
                word = values[0] # A word such as dog
                if word in unique_vocab: # Only take in account the words that appear in the SNLI dataset, to speed up training
                    vector = np.asarray(values[1:], "float32") # The embedding vector of the word
                    embeddings_dict[word] = vector # A dict key-value of word/vector
            except:
                # Some cases in glove seem to be messed up, such as . . . or at name@domain.com
                continue
    return embeddings_dict

def sentence_to_glove(tokens, glove_dict, dim=300):
    vectors = [glove_dict.get(token, np.zeros(dim)) for token in tokens]
    return np.array(vectors)

def encode_example(example, embeddings_dict):
    return {
        "premise_glove": sentence_to_glove(example["premise"], embeddings_dict),
        "hypothesis_glove": sentence_to_glove(example["hypothesis"], embeddings_dict),
        "label": example["label"]
    }

# The input to this, is normal sentences and the glove dict (so not glove vectors!)
def create_batch(tokenized_batch, glove_dict):
    # First sort the tokenized_batch in decreasing order
    lengths = np.fromiter((len(x) for x in tokenized_batch), dtype=int)
    longest_sentence = np.max(lengths)

    # Create a empty batch of zeros (to handle padding)
    embedded_batch = np.zeros((longest_sentence, len(tokenized_batch), 300))
    
    # Set the glove vector for each sentence into the embedded batch (and keep 0s as padding)
    for i, sentence in enumerate(tokenized_batch):
        glove_vector = sentence_to_glove(sentence, glove_dict)
        embedded_batch[:len(glove_vector), i, :] = glove_vector

    # The transpose here is done to follow batch_first, for the models this way the batch size comes first
    return torch.from_numpy(embedded_batch.transpose(1, 0, 2)).float(), lengths

# Input is a dataset split, such as the train set
def create_model_data_rep(dataset):
    # Get the train data ready
    data = {}
    data['premise'] = np.array(dataset['premise'], dtype=object)
    data['hypothesis'] = np.array(dataset['hypothesis'], dtype=object)
    data['label'] = np.array(dataset['label'], dtype=object)
    return data