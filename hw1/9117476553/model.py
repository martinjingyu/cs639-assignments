import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path, weights_only=False)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    
    vovab_size = len(vocab)
    emb = np.random.uniform(-0.25, 0.25, (vovab_size, emb_size)).astype(np.float32)
    
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.rstrip().split()
            if len(tokens) != emb_size + 1:
                continue
            word = tokens[0]
            if word in vocab.word2id:
                idx = vocab.word2id[word]
                emb[idx] = np.array(tokens[1:], dtype=np.float32)

    return emb



class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        args = self.args

        self.embedding = nn.Embedding(
            len(self.vocab),
            args.emb_size,
            padding_idx=self.vocab['<pad>']
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(args.hid_layer):
            in_size = args.emb_size if i == 0 else args.hid_size
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(in_size, args.hid_size),
                    nn.ReLU(),
                    nn.Dropout(args.hid_drop)
                )
            )

        self.output_layer = nn.Linear(args.hid_size, self.tag_size)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        v = 0.08
        
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.uniform_(param, -v, v)
            else:
                nn.init.zeros_(param)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight.data.copy_(torch.from_numpy(emb))

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        
        emb = self.embedding(x)  # [B, T, D]

        pad_id = self.vocab['<pad>']
        mask = (x != pad_id).float().unsqueeze(-1)  # [B, T, 1]
        emb = emb * mask

        if self.args.pooling_method == "sum":
            sent_emb = emb.sum(dim=1)

        elif self.args.pooling_method == "avg":
            sent_emb = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        elif self.args.pooling_method == "max":
            emb = emb.masked_fill(mask == 0, -1e9)
            sent_emb, _ = emb.max(dim=1)

        h = sent_emb
        for layer in self.hidden_layers:
            h = layer(h)

        scores = self.output_layer(h)
        return scores
