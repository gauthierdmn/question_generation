import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import sample_sequence


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 word_vectors,
                 bidirectional,
                 drop_prob=0.):
        super(Encoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.embedding = nn.Embedding.from_pretrained(word_vectors, freeze=True)
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=bidirectional,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        x = self.embedding(x)
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, (hidden, cell) = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)

        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)
        hidden = hidden[:, unsort_idx, :]
        cell = cell[:, unsort_idx, :]

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, word_vectors, n_layers, trg_vocab, device, dropout, attention=None,
                 min_len_sentence=5, max_len_sentence=50, top_k=0., top_p=0.9, temperature=0.7, greedy_decoding=False):
        super().__init__()
        self.output_dim = len(trg_vocab.itos)
        self.embedding = nn.Embedding.from_pretrained(word_vectors, freeze=True)
        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, bidirectional=False, dropout=dropout)
        self.attn = Attn(method="general", hidden_size=hidden_size, device=device) if attention else None
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.gen = Generator(decoder_size=hidden_size, output_dim=len(trg_vocab.itos))
        self.dropout = nn.Dropout(dropout)
        self.min_len_sentence = min_len_sentence
        self.max_len_sentence = max_len_sentence
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.greedy_decoding = greedy_decoding
        self.special_tokens_ids = [trg_vocab.stoi[t] for t in ["<EOS>", "<PAD>"]]
        self.device = device

    def forward(self, enc_out, enc_hidden, question=None):
        batch_size = enc_out.size(0)

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, 50, self.output_dim).to(self.device) if question is not None else []

        # TODO: we should have a "if bidirectional:" statement here, because does not work for unidirectional
        if isinstance(enc_hidden, tuple):  # meaning we have a LSTM encoder
            enc_hidden = tuple(
                (torch.cat((hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]), dim=2) for hidden in enc_hidden))
        else:  # GRU layer
            enc_hidden = torch.cat((enc_hidden[0:enc_hidden.size(0):2], enc_hidden[1:enc_hidden.size(0):2]), dim=2)

        enc_out = enc_out[:, -1, :].unsqueeze(1) if not self.attn else enc_out  # could use attention here
        dec_hidden = enc_hidden

        if question is not None:  # training with teacher
            for t in range(0, self.max_len_sentence):
                dec_input = question[:, t].unsqueeze(1)
                embedded = self.dropout(self.embedding(dec_input))  # (batch size, 1, emb dim)

                # Calculate attention weights and apply to encoder outputs
                attn_weights = self.attn(dec_hidden[0][-1], enc_out)
                context = attn_weights.bmm(enc_out)  # (B,1,V)

                dec_input = torch.cat((embedded, context), dim=2).float()
                if isinstance(self.rnn, nn.GRU):
                    dec_output, dec_hidden = self.rnn(dec_input, dec_hidden[0])
                else:
                    dec_output, dec_hidden = self.rnn(dec_input, dec_hidden)
                dec_output = self.dropout(dec_output)

                outputs[:, t, :] = self.gen(dec_output)
                # enc_out = dec_output
        else:  # eval
            dec_input = torch.zeros(enc_out.size(0), 1).fill_(2).long().to(self.device)
            for t in range(0, self.max_len_sentence):
                embedded = self.dropout(self.embedding(dec_input))  # (batch size, 1, emb dim)

                # Calculate attention weights and apply to encoder outputs
                attn_weights = self.attn(dec_hidden[0][-1], enc_out)
                context = attn_weights.bmm(enc_out)  # (B,1,V)

                dec_input = torch.cat((embedded, context), dim=2).float()
                if isinstance(self.rnn, nn.GRU):
                    dec_output, dec_hidden = self.rnn(dec_input, dec_hidden[0])
                else:
                    dec_output, dec_hidden = self.rnn(dec_input, dec_hidden)
                    dec_output = self.dropout(dec_output)
                out = self.gen(dec_output)

                out, probs = sample_sequence(out, self.top_k, self.top_p, self.temperature, self.greedy_decoding)
                if t < self.min_len_sentence and out.item() in self.special_tokens_ids:
                    while out.item() in self.special_tokens_ids:
                        out = torch.multinomial(probs, num_samples=1)

                if out.item() in self.special_tokens_ids:
                    break
                outputs.append(out.item())
                dec_input = out.long().unsqueeze(1)

        return outputs


class Generator(nn.Module):
    def __init__(self, decoder_size, output_dim):
        super(Generator, self).__init__()
        self.gen_func = nn.LogSoftmax(dim=-1)
        self.generator = nn.Linear(decoder_size, output_dim)

    def forward(self, x):
        out = self.gen_func(self.generator(x)).squeeze(1)
        return out


class Attn(nn.Module):
    def __init__(self, method, hidden_size, device):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.device = device

    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        attn_energies = self.score(H, encoder_outputs)  # compute attention score

        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = torch.ByteTensor(mask).unsqueeze(1).to(self.device)  # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)

        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]

        return energy.squeeze(1)  # [B*T]
