# Modified from https://github.com/xsway/language-models.
# Classes to use: RNNLanguageModel or BidirectionalRNNLanguageModel.
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Input word embeddings."""
    def __init__(self, vocab_size, emb_size, fixed_embs=False):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        return self.embedding(input)


class Decoder(nn.Module):
    """Output word embeddings."""
    def __init__(self, vocab_size, hidden_size, tie_weights="", encoder=None):
        super(Decoder, self).__init__()

        self.linear = None
        if tie_weights == "standard":
            if hidden_size != encoder.emb_size:
                raise ValueError('When using the tied flag, hidden size of last RNN layer must be equal to emb size')
            self.decoder = nn.Linear(encoder.emb_size, vocab_size, bias=False)
            self.decoder.weight = encoder.embedding.weight
        elif tie_weights == "plusL":
            self.linear = nn.Linear(hidden_size, encoder.emb_size, bias=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)
            self.decoder = nn.Linear(encoder.emb_size, vocab_size, bias=False)
            self.decoder.weight = encoder.embedding.weight
        else:
            # no tying
            self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
            self.init_weights()

    def forward(self, input):
        if self.linear:
            input = self.linear(input)
        return self.decoder(input)

    def init_weights(self):
        initrange = 0.1
        if self.linear:
            self.linear.weight.data.uniform_(-initrange / 10, initrange / 10)
        self.decoder.weight.data.uniform_(-initrange, initrange)


class StackedRNN(nn.Module):
    """Stacked RNN module. Note: can instead use torch LSTM for stacked RNNs."""
    def __init__(self, rnn_type, input_size, hidden_sizes, dropout):
        super(StackedRNN, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.output_size = hidden_sizes[-1]

        self.nlayers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()

        self.drop = nn.Dropout(dropout)

        sizes = [input_size, ] + hidden_sizes

        for i in range(1, self.nlayers + 1):
            input_size = sizes[i-1]
            hidden_size = sizes[i]

            if rnn_type in ['LSTM', 'GRU']:
                rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1)
            else:
                try:
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                except KeyError:
                    raise ValueError("""An invalid option for `--rnncell` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
                rnn = nn.RNN(input_size, hidden_size, 1, nonlinearity=nonlinearity)
            self.layers.append(rnn)

    def forward(self, input, hidden):
        # Input shape batch_size x seq_len x emb_dim.
        # Permute so that seq_len is first.
        input = input.permute(1,0,2)

        for rnn, h in zip(self.layers, hidden):
            rnn.flatten_parameters() # Save memory by flattening.
            output, _ = rnn(input, h)
            input = self.drop(output)

        output = output.permute(1,0,2) # Return to batch dimension first.
        return output

    def init_hidden(self, batch_size, input):
        # Note: input is only used to get the tensor dtype and correct GPU device.
        hidden = []
        for nhid in self.hidden_sizes:
            if self.rnn_type == 'LSTM':
                hidden.append((input.new(1, batch_size, nhid).fill_(0.01),
                               input.new(1, batch_size, nhid).fill_(0.01)))
            else:
                hidden.append(input.new(1, batch_size, nhid).fill_(0.01))
        return hidden


class RNNLanguageModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, vocab_size, emb_size, hidden_sizes, dropout,
                 rnn_type="LSTM", fixed_embs=False, tied=None):
        super(RNNLanguageModel, self).__init__()

        self.encoder = Encoder(vocab_size, emb_size, fixed_embs)
        self.decoder = Decoder(vocab_size, hidden_sizes[-1], tied, self.encoder)
        self.criterion = nn.CrossEntropyLoss()

        self.rnn = StackedRNN(rnn_type, emb_size, hidden_sizes, dropout)
        self.drop = nn.Dropout(dropout)

    @classmethod
    def from_config(cls, vocab_size, config):
         # Initialize from config.
         return cls(vocab_size, config["embedding_size"], config["hidden_sizes"],
                    config["dropout"], config["rnn_cell"], tied=config["emb_tied"])

    def forward(self, input, labels, hidden, loss_only=True):
        # Input batch_size x seq_len.
        # Note that if provided, initial hidden states should have shape:
        # 1 x batch_size x hidden_dim.
        emb = self.drop(self.encoder(input))
        if hidden is None: # Initial hidden state.
            hidden = self.init_hidden(input.shape[0], emb)
        output = self.rnn(emb, hidden)
        output = self.drop(output)

        decoded = self.decoder(output.reshape(output.size(0) * output.size(1), output.size(2)))
        loss = self.criterion(decoded, labels.flatten())
        # Batch_size, seq_len vocab_size.
        decoded = decoded.reshape(output.size(0), output.size(1), decoded.size(-1))
        if loss_only:
            return (loss, None)
        else:
            return (loss, decoded)

    def init_hidden(self, batch_size, input):
        return self.rnn.init_hidden(batch_size, input)


class BidirectionalRNNLanguageModel(nn.Module):
    """Container module with a (shared) encoder, two recurrent modules -- forward and backward -- and a decoder."""

    def __init__(self, vocab_size, emb_size, hidden_sizes, dropout,
                 rnn_type="LSTM", fixed_embs=False, tied=None):
        super(BidirectionalRNNLanguageModel, self).__init__()
        self.drop = nn.Dropout(dropout)

        self.encoder = Encoder(vocab_size, emb_size, fixed_embs)
        self.decoder = Decoder(vocab_size, hidden_sizes[-1], tied, self.encoder)
        self.criterion = nn.CrossEntropyLoss()

        self.forward_lstm = StackedRNN(rnn_type, emb_size, hidden_sizes, dropout)
        self.backward_lstm = StackedRNN(rnn_type, emb_size, hidden_sizes, dropout)

        self.rnn_type = rnn_type
        self.hidden_sizes = hidden_sizes
        self.nlayers = len(hidden_sizes)

    @classmethod
    def from_config(cls, vocab_size, config):
         # Initialize from config.
         return cls(vocab_size, config["embedding_size"], config["hidden_sizes"],
                    config["dropout"], config["rnn_cell"], tied=config["emb_tied"])

    def forward(self, input, labels, hidden, loss_only=True):
        # Inputs batch_size x seq_len. The backward input should not be flipped yet.
        # For example inputs should be ([0,1,2], [2,3,4]) corresponding to:
        # Input:   [0 1 2]
        # Labels:    [1 2 3]
        # Backward:    [2 3 4]
        # Note that if provided, initial hidden states should have shape:
        # 1 x batch_size x hidden_dim.
        input_f, input_b = input
        input_b = torch.flip(input_b, dims=[1])
        emb_f = self.drop(self.encoder(input_f))
        emb_b = self.drop(self.encoder(input_b))

        if hidden is None: # Initial hidden states.
            # Note: can just use forward inputs and embeddings because they are
            # only used to determine batch size, dtype, and GPU device.
            hidden = self.init_hidden(input_f.shape[0], emb_f)
        hidden_f = hidden[0]
        hidden_b = hidden[1]

        output_f = self.forward_lstm(emb_f, hidden_f)
        output_b = self.backward_lstm(emb_b, hidden_b)

        # Output is sum of forward and backward.
        # Flip back along dimension 1 (sequence length).
        output = output_f + torch.flip(output_b, dims=[1])

        decoded = self.decoder(output.reshape(output.size(0)*output.size(1), output.size(2)))
        loss = self.criterion(decoded, labels.flatten())
        # Batch_size, seq_len vocab_size.
        decoded = decoded.reshape(output.size(0), output.size(1), decoded.size(-1))
        if loss_only:
            return (loss, None)
        else:
            return (loss, decoded)

    def init_hidden(self, batch_size, input):
        return (self.forward_lstm.init_hidden(batch_size, input),
            self.backward_lstm.init_hidden(batch_size, input))
