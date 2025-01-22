import torch
import torch.nn as nn


class EncoderLstm(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, dropout_prob=0, layer_norm=False,
                 num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size * 2) if layer_norm else nn.Identity()
        self.fc_mean = nn.Linear(hidden_size * 2, latent_size * 2)
        self.fc_logvar = nn.Linear(hidden_size * 2, latent_size * 2)

    def forward(self, x):
        # Reversing the sequence of indices
        x = torch.flip(x, [1])
        embedded = self.embedding(x)

        outputs, (hidden, cell) = self.lstm(embedded)
        hidden, cell = hidden.contiguous(), cell.contiguous()
        concatenated_states = torch.cat((hidden, cell), dim=-1)
        concatenated_states = self.layer_norm(concatenated_states)
        concatenated_states = self.dropout(concatenated_states)
        mean = self.fc_mean(concatenated_states)
        log_var = self.fc_logvar(concatenated_states)
        return outputs, mean, log_var


class DecoderLstm(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, dropout_prob=0.2, layer_norm=False,
                 num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, latent_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(latent_size, vocab_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(latent_size) if layer_norm else nn.Identity()

    def forward(self, x, z):
        out = self.embedding(x)
        hidden, cell = torch.chunk(z, chunks=2, dim=-1)
        hidden, cell = hidden.contiguous(), cell.contiguous()
        out, (hidden, cell) = self.lstm(out, (hidden, cell))
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc(out).reshape(out.size(0), -1)
        hidden_cell = torch.cat((hidden, cell), dim=-1)
        return out, hidden_cell


class FFNN(nn.Module):
    def __init__(self, input_dim, dropout_prob=0, layer_norm=False):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(256) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(64) if layer_norm else nn.Identity()

    def forward(self, x):
        x = torch.relu(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
