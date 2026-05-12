
import torch, nltk, pickle
from torch import nn
from collections import Counter
from tqdm import tqdm
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os
from nltk import word_tokenize
nltk.download('punkt_tab')

###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in word_tokenize(text)]

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """

    # TODO: build the vocabulary, possibly truncating it to max_voc_size if that is specified.
    # Then return a tokenizer object (implemented below).
    f = open(train_file, 'r').read()
    counter = Counter(tokenize_fun(f))
    # sorted_tokens = list(counter.keys())
    sorted_tokens = [tok for tok, _ in counter.most_common()]

    # sorted_tokens.sort(key=lambda x: x[0], reverse=True)
    # print(sorted_tokens[:10])
    # sorted_tokens  also cound use counter.most_common() 
    vocab = {}
    sorted_tokens = [bos_token, eos_token, unk_token, pad_token] + sorted_tokens
    if max_voc_size is not None:
        sorted_tokens = sorted_tokens[:max_voc_size]
    if sorted_tokens is not None:
        i = 0
        while i < len(sorted_tokens): # and i < model_max_length:
            vocab[sorted_tokens[i]] = i
            i += 1
    return A1Tokenizer(vocab, model_max_length, pad_token_id=vocab[pad_token])

class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, vocab, model_max_length, pad_token_id):
        # TODO: store all values you need in order to implement __call__ below.
        self.pad_token_id = pad_token_id     # Compulsory attribute.
        self.model_max_length = model_max_length # Needed for truncation.
        self.vocab = vocab

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize. [LIST OF STRINGS!]
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        # tokenizer takes list of lists of strings  like this test_texts = [['This is a test.', 'Another test.']]
        tok_texts = []
        attention_masks = []
        max_len = 0
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        for text in texts:
            split_text = lowercase_tokenizer(text)

            tok_text = [self.vocab['<BOS>']] + [self.vocab.get(t, self.vocab.get('<UNK>')) for t in split_text] \
                + [self.vocab["<EOS>"]]
            tok_text = tok_text[:self.model_max_length] if (truncation and self.model_max_length) else tok_text
            max_len = max(max_len, len(tok_text))
            attention_mask = [1 if t != self.pad_token_id else 0 for t in tok_text]
            tok_texts.append(tok_text)
            attention_masks.append(attention_mask)       
        if padding: # to the max length ? no info about what is the desire size 
            for i in range(len(tok_texts)):
                while len(tok_texts[i]) < max_len:
                    tok_texts[i].append(self.pad_token_id)
                    attention_masks[i].append(0)
        if return_tensors == 'pt':
            tok_texts = torch.stack([torch.tensor(tok_text) for tok_text in tok_texts])
            attention_masks = torch.stack([torch.tensor(attention_mask) for attention_mask in attention_masks])
        return BatchEncoding({'input_ids': tok_texts, 'attention_mask': attention_masks})

    
        # TODO: Your work here is to split the texts into words and map them to integer values.
        # 
        # - If `truncation` is set to True, the length of the encoded sequences should be 
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.
        # return BatchEncoding({'input_ids': ...})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.vocab)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=0, embedding_size=0, hidden_size=0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        # self.rnn = nn.GRU(config.embedding_size, config.hidden_size, batch_first=True)
        self.rnn = nn.GRU(config.embedding_size, config.hidden_size, batch_first=True, num_layers=2, dropout=0.3) # mroe compelx for better resutls 
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)

        # Note: -100 is the value HuggingFace conventionally uses to refer to tokens
        # where we do not want to compute the loss.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)


    def forward(self, input_ids, labels=None):
        """The forward pass of the RNN-based language model.
        
           Args:
             - input_ids:  The input tensor (2D), consisting of a batch of integer-encoded texts.
             - labels:     The reference tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             A CausalLMOutput containing
               - logits:   The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
               - loss:     The loss computed on this batch.               
        """
        embedded = self.embedding(input_ids)
        rnn_out, _ = self.rnn(embedded)
        logits = self.unembedding(rnn_out)
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_func(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
)
            # loss = self.loss_func(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return CausalLMOutput(logits=logits, loss=loss)


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.
def calculate_perplexity(model, eval_dataset, tokenizer, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in tqdm(eval_dataset, desc='Perplexity eval', unit='batch'):
        with torch.no_grad():
            input_ids = tokenizer(batch, truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(input_ids=input_ids, labels=labels)
            # from hugging faces doc
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            n_tokens = (labels[:, 1:] != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)
    print(f'Validation Perplexity: {perplexity:.4f}')

# another evaluation - find n nearest neighbors of a given word in the embedding space, and output these n words using cosine similarity
def find_nearest_neighbors(model, tokenizer, word, n=5):
    if word not in tokenizer.vocab:
        print(f'Word "{word}" not found in vocabulary.')
        return
    word_id = tokenizer.vocab[word]
    word_embedding = model.embedding.weight[word_id].detach().cpu().numpy()
    embeddings = model.embedding.weight.detach().cpu().numpy()
    cosine_similarities = np.dot(embeddings, word_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(word_embedding))
    nearest_indices = np.argsort(cosine_similarities)[-n-1:-1][::-1]  # Exclude the word itself
    nearest_words = [list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(idx)] for idx in nearest_indices]
    print(f'Nearest neighbors of "{word}": {nearest_words}')

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        # if torch.mps.is_available():
        #     return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        # loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_train_epochs)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(
            self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            self.eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False
        )

        # TODO: Your work here is to implement the training loop.
        #       
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       input_ids = apply your tokenizer to B
        #       labels = input_ids with padding replaced by -100
	    #       put input_ids and labels onto the GPU (or whatever device you use)
        #       apply the model to input_ids and labels
        #       get the loss from the model output
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward()
        #       optimizer.step()
        # for tqdm
        print('Starting training...')
        # calculate_perplexity(self.model, self.eval_dataset, self.tokenizer, device=device) # ie does another eval loop but fine 
        # find_nearest_neighbors(self.model, self.tokenizer, 'sweden', n=5)
        for epoch in range(args.num_train_epochs):
            self.model.train()
            print(f'Training Epoch {epoch + 1}/{args.num_train_epochs}')
            for batch in tqdm(train_loader, desc=f'Train epoch {epoch + 1}/{args.num_train_epochs}', unit='batch'):
                input_ids = self.tokenizer(batch, truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                loss = self.model(input_ids=input_ids, labels=labels).loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            self.model.eval()
            total_loss = 0
            if epoch % 5 == 0: # or if you want to do it every few epochs
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f'Validate epoch {epoch + 1}/{args.num_train_epochs}', unit='batch'):
                        input_ids = self.tokenizer(batch, truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
                        labels = input_ids.clone()
                        labels[labels == self.tokenizer.pad_token_id] = -100
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        total_loss += outputs.loss.item() * input_ids.size(0)
            scheduler.step()
            if epoch % 2 == 0: # or if you want to do it every few epochs
                calculate_perplexity(self.model, self.eval_dataset, self.tokenizer, device=device) # ie does another eval loop but fine 
                find_nearest_neighbors(self.model, self.tokenizer, 'sweden', n=5)
            avg_loss = total_loss / len(self.eval_dataset)
            print(f'Epoch {epoch + 1}/{args.num_train_epochs}, Validation Loss: {avg_loss:.4f}')
        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)

    
