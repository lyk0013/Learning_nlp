import time
import random
from collections import Counter
import torch
from torch import nn, optim
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.datasets import IMDB
import torchtext.vocab as Vocab
from torchtext.data.utils import get_tokenizer


def get_vocab_imdb(data, min_freq=5):
    return get_vocab(data, min_freq)

def get_vocab(data, min_freq=5):
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    counter = Counter()
    for (label, line) in data:
        counter.update(tokenizer(line))
    vocab = Vocab.vocab(counter, min_freq=5)
    unk_token = '<unk>'
    vocab.insert_token(unk_token, 0)
    vocab.set_default_index(vocab[unk_token])
    return vocab
    
    
def get_tokenized_imdb(data):
    return get_tokenized(data)

def get_tokenized(data):
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    return [tokenizer(review) for (_, review) in data]


def preprocess(data, vocab, max_l=500):
    
    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l-len(x))
    tokenized_data = get_tokenized(data)
    features = torch.tensor(
        [pad(vocab.lookup_indices(words)) for words in tokenized_data]
    )
    labels = torch.tensor([1 if score=='pos' else 0 for (score, _) in data])
    return features, labels
    
    
def preprocess_imdb(data, vocab, max_l=500):
    return preprocess(data, vocab, max_l)
    
    
def train(model, iterator, optimizer, loss, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    model.train()
    for X, y in iterator:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        predictions = model(X).squeeze(0)
        l = loss(predictions, y)
        acc = (predictions.argmax(dim=1) == y).sum().cpu().item()/X.shape[0]
        l.backward()
        optimizer.step()
        epoch_loss += l.cpu().item()
        epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc/len(iterator)
    
    
def evaluate(model, iterator, loss, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    model.eval()  # ??????dropout??????????????????batch normalization
    with torch.no_grad():  # ???????????????????????????????????????
        for X, y in iterator:
            X = X.to(device)
            y = y.to(device)
            predictions = model(X).squeeze(0)
            l = loss(predictions, y)
            acc = (predictions.argmax(dim=1) == y).sum().cpu().item()/X.shape[0]

            epoch_loss += l.cpu().item()
            epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc/len(iterator)
    
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs