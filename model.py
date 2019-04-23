import torch
import torch.nn as nn
import random

import layers


class Seq2Seq(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, output_dim, device, drop_prob=0.):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size * 2,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.dec = layers.AttnDecoder(input_size=hidden_size,
                                      output_dim=output_dim,
                                      word_vectors=word_vectors,
                                      hidden_size=2 * hidden_size,
                                      n_layers=1,
                                      dropout=drop_prob)

        self.device = device

    def forward(self, sw_idxs, sc_idxs, qw_idxs, teacher_forcing_ratio=0.5):
        batch_size = sw_idxs.size(0)
        max_len = qw_idxs.size(1)

        s_mask = torch.zeros_like(sw_idxs) != sw_idxs
        s_len = s_mask.sum(-1)

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, self.output_dim).to(self.device)

        s_emb = self.emb(sw_idxs, sc_idxs)  # (batch_size, s_len, hidden_size)

        s_enc, (hidden, cell) = self.enc(s_emb, s_len)  # (batch_size, s_len, 2 * hidden_size)

        hidden = torch.cat((hidden[-1], hidden[-2]), dim=1).unsqueeze(0)
        cell = torch.cat((cell[-1], cell[-2]), dim=1).unsqueeze(0)

        dec_input = qw_idxs[:, 0]

        for t in range(1, max_len):
            output, hidden, cell = self.dec(dec_input, s_enc, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]  # (batch_size)
            dec_input = (qw_idxs[:, t] if teacher_force else top1)

        return outputs


class BiDAF(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, output_dim, device, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=2 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.dec_emb = layers.DecEmbedding(word_vectors=word_vectors,
                                           hidden_size=hidden_size,
                                           drop_prob=drop_prob)

        self.dec = layers.Decoder(input_size=2 * hidden_size,
                                  output_dim=output_dim,
                                  word_vectors=word_vectors,
                                  hidden_size=2 * hidden_size,
                                  n_layers=1,
                                  dropout=drop_prob)

        self.device = device

    def forward(self, sw_idxs, sc_idxs, aw_idxs, ac_idxs, qw_idxs, teacher_forcing_ratio=0.5):
        batch_size = sw_idxs.size(0)
        max_len = qw_idxs.size(1)

        s_mask = torch.zeros_like(sw_idxs) != sw_idxs
        a_mask = torch.zeros_like(aw_idxs) != aw_idxs
        s_len, a_len = s_mask.sum(-1), a_mask.sum(-1)

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, self.output_dim).to(self.device)

        s_emb = self.emb(sw_idxs, sc_idxs)   # (batch_size, s_len, hidden_size)
        a_emb = self.emb(aw_idxs, ac_idxs)   # (batch_size, a_len, hidden_size)

        s_enc, _ = self.enc(s_emb, s_len)   # (batch_size, s_len, 2 * hidden_size)
        a_enc, _ = self.enc(a_emb, a_len)   # (batch_size, a_len, 2 * hidden_size)

        att = self.att(s_enc, a_enc, s_mask, a_mask)   # (batch_size, s_len, hidden_size)

        _, (hidden, cell) = self.mod(att, s_len)

        # we add the forward and backward hiddens and cells of the top LSTM layer to use for the decoder
        enc_hidden = torch.add(hidden[-1], hidden[-2]).unsqueeze(0)
        #enc_cell = torch.add(cell[-1], cell[-2]).unsqueeze(0)

        hidden = torch.cat((hidden[-1], hidden[-2]), dim=1).unsqueeze(0)
        cell = torch.cat((cell[-1], cell[-2]), dim=1).unsqueeze(0)

        dec_input = qw_idxs[:, 0]

        for t in range(1, max_len):
            output, hidden, cell = self.dec(dec_input, enc_hidden, hidden, cell)   # output, hidden, cell
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]   # (batch_size)
            dec_input = (qw_idxs[:, t] if teacher_force else top1)

        return outputs
