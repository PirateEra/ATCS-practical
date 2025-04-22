import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from datasets import load_dataset
import nltk
from utils import preprocess_data, build_vocab, create_embedding_dict, encode_example, create_model_data_rep, create_batch
from torch.utils.tensorboard import SummaryWriter
import os

class UnidirectionalLSTM(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=512, dropout=0.2):
        super(UnidirectionalLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim,  # input dimension (word embedding dimension, which is 300 for GloVe)
                            hidden_size=hidden_dim, # hiddendim (the dimension of the lstm itself)
                            num_layers=1,
                            batch_first=True,        # batch size comes first in the input of x
                            dropout=dropout,
                            bidirectional=False)

    # We assume that x are already padded and sorted glove embeddings for the sequences in a batch
    def forward(self, x):
        device = self.lstm.weight_ih_l0.device
        embeddings, lengths = x

        embeddings = embeddings.to(device) # Embeddings comes from most likely the cpu if it was made using a util function, so set it to device to be sure
        packed_input = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False).to(device)
        # forward pass
        _, (n_hidden, _) = self.lstm(packed_input)
        
        # # hidden state of the last LSTM layer
        embedding = n_hidden[-1]
        
        return embedding

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=512, dropout=0.2, max_pooling=False):
        super(BidirectionalLSTM, self).__init__()
        self.max_pooling = max_pooling
        self.lstm = nn.LSTM(input_size=input_dim,  # input dimension (word embedding dimension, which is 300 for GloVe)
                            hidden_size=hidden_dim, # hiddendim (the dimension of the lstm itself)
                            num_layers=1,
                            batch_first=True,        # batch size comes first in the input of x
                            dropout=dropout,
                            bidirectional=True)

    # We assume that x are already padded and sorted glove embeddings for the sequences in a batch
    def forward(self, x):
        device = self.lstm.weight_ih_l0.device
        embeddings, lengths = x
        embeddings.to(device) # Embeddings comes from most likely the cpu if it was made using a util function, so set it to device to be sure
        packed_input = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False).to(device)
        # forward pass
        output, (n_hidden, _) = self.lstm(packed_input)
        
        if self.max_pooling:
            # unpack the output
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            # Prepare the list to store max-pooled embeddings
            max_embeddings = [torch.max(i, 0)[0] for i in output]
            # Stack the max-pooled embeddings to form a batch tensor
            embedding = torch.stack(max_embeddings, 0)

        else:
            # hidden state of the bidirectional lstm, we now concat since we go both directions and we need to use both
            embedding = torch.cat((n_hidden[-2], n_hidden[-1]), dim=-1)
        
        return embedding

class MeanEncoder():
    def __init__(self, device):
        self.device = device

    def __call__(self, x):
        embeddings, _ = x
        embeddings = embeddings.mean(dim=(1)).to(self.device)
        return embeddings

class SNLIClassifier(nn.Module):
    def __init__(self, params, device, input_dim=512, class_count=3, connected_dim=512):
        super(SNLIClassifier, self).__init__()
        
        #-------
        # Setup the model
        #-------
        self.device = device
        self.setup_encoder(params)
        # 4 * input_dim due to concatenation, 512 as connected_dim can be a design choice based on the paper (section 3.3)
        input_dim = 4 * input_dim
        if params.encoder_model == 'bi-lstm' or params.encoder_model == 'bi-max-lstm':
            input_dim = input_dim * 2 # *2 because it goes both directions so its twice the size
        if params.encoder_model == 'mean':
            input_dim = 1200 # This would be 4*300, since we take the mean we get one average embedding vector (based on Glove) this would be 300
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, connected_dim),
            #nn.ReLU(),
            nn.Linear(connected_dim, connected_dim),
            #nn.ReLU(),
            nn.Linear(connected_dim, class_count)
        )
    

    # Minibatches of size 64 are used
    def forward(self, premise_batch, hypothesis_batch):
        # Embed the premise and hypothesis
        u = self.encoder(premise_batch)
        v = self.encoder(hypothesis_batch)
        # u: [batch_size, 300], v: [batch_size, 300], both embedded/encoded by a encoder such as an LSTM

        # Combine the embeddings just like in the conneau et al paper
        features = torch.cat([u,v, torch.abs(u - v), u * v], dim=1)  # dim=1 for batch mode

        logits = self.classifier(features)

        return logits  # Softmax will still need to be applied to this! 3 classes (entailment, contradiction, neutral)

    def encode(self, batch):
        return self.encoder(batch)

    def setup_encoder(self, params):
        type = params.encoder_model
        if type == 'uni-lstm':
            self.encoder = UnidirectionalLSTM(
                300,
                params.lstm_dim,
                dropout=params.dropout_encoder
            )
        elif type == 'bi-lstm':
            self.encoder = BidirectionalLSTM(
                300,
                params.lstm_dim,
                dropout=params.dropout_encoder,
                max_pooling=False
            )
        elif type == 'bi-max-lstm':
            self.encoder = BidirectionalLSTM(
                300,
                params.lstm_dim,
                dropout=params.dropout_encoder,
                max_pooling=True
            )
        elif type == 'mean':
            self.encoder = MeanEncoder(self.device)

        # Move encoder to gpu
        if type != 'mean':
            self.encoder = self.encoder.to(self.device)

class SNLIModel:
    def __init__(self, params):
        self.params = params
        # Setup/imports
        print("Setting up all imports and downloads")
        self.writer = SummaryWriter(log_dir=f'runs/{params.encoder_model}')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nltk.download('punkt') # this one was needed for the tokenizer of nltk (pretrained model, to properly tokenize aka punkt)
        nltk.download('punkt_tab')
        self.dataset = load_dataset("stanfordnlp/snli")
        # Seeds
        np.random.seed(self.params.seed)
        torch.manual_seed(self.params.seed)
        torch.cuda.manual_seed(self.params.seed)
        # Preprocessing
        print("Preprocessing the data")
        self.preprocessed_data = preprocess_data(self.dataset)
        self.unique_vocab = build_vocab(self.preprocessed_data)
        self.glove_dict = create_embedding_dict(self.unique_vocab, self.params.glovepath)
        # Setting up the models
        print("Setting up the classifier (and the encoder within)")
        #
        self.classifier = SNLIClassifier(self.params, self.device, input_dim=self.params.lstm_dim, class_count=self.params.class_count, connected_dim=self.params.classifier_dim)
        self.classifier = self.classifier.to(self.device)
        # Setting up the optimizer and loss
        print("Setting up the optimizer and loss function")
        class_weights = torch.FloatTensor(self.params.class_count).fill_(1) # The class weights for the loss function (start at 1.0)
        self.loss_function = nn.CrossEntropyLoss(weight=class_weights) # Use cross entropyloss for the loss function
        self.loss_function.size_average = False # Better because we are using a small batch size such as 64
        self.loss_function = self.loss_function.to(self.device)
        # optimizer
        self.optimizer = optim.SGD(self.classifier.parameters(), lr=self.params.sgd_lr, momentum=self.params.sgd_momentum)

        # Get the train data ready
        #self.train_data = create_model_data_rep(self.preprocessed_data["train"].select(range(5000)))
        self.train_data = create_model_data_rep(self.preprocessed_data["train"])

        # Get the dev data ready since the paper does the following: 
        # At each epoch, we divide the learning rate by 5 if the dev accuracy decreases.
        self.dev_data = create_model_data_rep(self.preprocessed_data["validation"])
        self.previous_dev_accuracy = float('-inf')
        self.best_dev_accuracy = 0

        # For if testing needs to be done, use evaluate_accuracy(self.test_data)
        self.test_data = create_model_data_rep(self.preprocessed_data["test"])

        # If a checkpoint path is given, load the checkpoint
        if self.params.checkpoint_path != 'none':
            self.load_checkpoint()

    def train_epoch(self):
        print(f"Training epoch {self.epoch}")
        batch_size = self.params.batch_size
        self.classifier.train() # Set the classifier to train mode
        train_rows = len(self.train_data['premise'])

        # Shuffle the data and get the premises, hypothesis and target_labels (shuffled) for the entire training set
        random_indices = np.random.permutation(train_rows)
        premises = self.train_data['premise'][random_indices]
        hypothesis = self.train_data['hypothesis'][random_indices]
        target_labels = self.train_data['label'][random_indices]

        # Decay, like in the note in the assignment (0.99 like in the paper)
        self.optimizer.param_groups[0]['lr'] *= self.params.decay

        # Go through every batch
        total_correct = 0
        total_loss = 0
        for index in range(0, train_rows, batch_size):
            # Create batches for the premise and hypothesis, and the target label
            premise_batch = create_batch(premises[index:index+batch_size], self.glove_dict)
            hypothesis_batch = create_batch(hypothesis[index:index+batch_size], self.glove_dict)
            target_batch = torch.LongTensor(target_labels[index:index+batch_size].astype(int)).to(self.device) # Has to be LongTensor for the Crossentropy module

            # Forward the classifier model, using the batches
            logits = self.classifier(premise_batch, hypothesis_batch)

            # Get the amount of correctly predicted labels, u take the argmax of the logits which is the most likely label in this case
            predictions = logits.argmax(dim=1).to(self.device)
            correct_predictions = (predictions == target_batch).sum().cpu()
            total_correct += correct_predictions

            # Update the loss
            loss = self.loss_function(logits, target_batch)
            total_loss += loss.item()
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
        
        # Values to keep track of during epochs
        accuracy = total_correct / train_rows
        avg_loss = total_loss / (train_rows / batch_size)
        print(f"The avg loss of epoch {self.epoch} is {avg_loss}")
        learning_rate = self.optimizer.param_groups[0]['lr']
        # Tensorboard logging
        self.writer.add_scalar('Training Loss', avg_loss, self.epoch)
        self.writer.add_scalar('Training Accuracy', accuracy, self.epoch)
        self.writer.add_scalar('Learning Rate', learning_rate, self.epoch)
        self.writer.flush()

        dev_accuracy = self.evaluate_accuracy(self.dev_data)
        print(f"The current dev accuracy is {dev_accuracy}")
        # If the dev accuracy is below what it just was in the previous epoch, divide LR by 5
        if dev_accuracy <= self.previous_dev_accuracy:
            self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] / 5 # Divide by 5
            print(f"dividing the LR by 5, due to a low dev accuracy, {dev_accuracy} compared to previously being {self.previous_dev_accuracy}")
        self.previous_dev_accuracy = dev_accuracy
        
        # Save checkpoint if the model is better than the previous epoch model
        if dev_accuracy > self.best_dev_accuracy:
            self.best_dev_accuracy = dev_accuracy
            self.save_checkpoint()

        # Return this, for the training loop in train(), to potentially prematurely stop, keep in mind i return the non divided LR here
        # So the divided LR will be returned the epoch AFTER that
        return learning_rate


    def train(self):
        print("Starting training!")
        # Start training
        self.epoch = 1
        current_lr = self.params.sgd_lr
        while self.epoch <= self.params.epochs:
            current_lr = self.train_epoch()
            if current_lr < 1e-5:
                print(f"Stopping early at epoch {self.epoch} because the current learning rate {current_lr:.2e} is below 1e-5")
                break
            self.epoch+=1
    
    def evaluate_accuracy(self, dataset):
        self.classifier.eval()
        batch_size = self.params.batch_size
        train_rows = len(dataset['premise'])
        # Go through every batch
        total_correct = 0
        for index in range(0, train_rows, batch_size):
            # Create batches for the premise and hypothesis, and the target label
            premise_batch = create_batch(dataset['premise'][index:index+batch_size], self.glove_dict)
            hypothesis_batch = create_batch(dataset['hypothesis'][index:index+batch_size], self.glove_dict)
            target_batch = torch.LongTensor(dataset['label'][index:index+batch_size].astype(int)).to(self.device)

            # Forward the classifier model, using the batches
            logits = self.classifier(premise_batch, hypothesis_batch)

            # Get the amount of correctly predicted labels
            predictions = logits.argmax(dim=1).to(self.device)
            correct_predictions = (predictions == target_batch).sum().cpu()
            total_correct += correct_predictions

        accuracy = total_correct / train_rows
        return accuracy

    def predict(self, premise, hypothesis):
        # These can even be batches of 1, for singular examples
        premise_batch = create_batch(premise, self.glove_dict)
        hypothesis_batch = create_batch(hypothesis, self.glove_dict)
        logits = self.classifier(premise_batch, hypothesis_batch)
        predictions = logits.argmax(dim=1).to(self.device)
        map_list = ['entailment', 'neutral', 'contradiction']
        prediction_texts = [map_list[pred] for pred in predictions]
        return predictions, prediction_texts

    def save_checkpoint(self):
        if not os.path.exists(self.params.outputdir):
            os.makedirs(self.params.outputdir)

        checkpoint = {
            'model_state_dict': self.classifier.state_dict(), # This saves everything, so it also saved the encoder within the classifier
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'dev_accuracy': self.best_dev_accuracy,
        }

        output_path = os.path.join(self.params.outputdir, self.params.outputmodelname+".pth")
        torch.save(checkpoint, output_path)
        print(f"Checkpoint saved at epoch {self.epoch} with dev accuracy {self.best_dev_accuracy:.4f}, modelname would be {self.params.outputmodelname}")
    
    def load_checkpoint(self):
        checkpoint_path = self.params.checkpoint_path
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.epoch = checkpoint['epoch']
            self.best_dev_accuracy = checkpoint['dev_accuracy']
            
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"Checkpoint {checkpoint_path} not found!")