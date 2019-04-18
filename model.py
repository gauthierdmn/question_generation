import torch
import torch.nn as nn
import random

import layers


class Seq2Seq(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, device, drop_prob=0.):
        super().__init__()

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.encoder = layers.RNNEncoder(input_size=hidden_size * 2,
                                         hidden_size=hidden_size,
                                         num_layers=1,
                                         drop_prob=drop_prob)

        self.decoder = layers.Decoder(output_dim=40000,
                                      word_vectors=word_vectors,
                                      hidden_size=hidden_size,
                                      n_layers=1,
                                      dropout=drop_prob)

        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [batch size, src sent len]
        # trg = [batch size, trg sent len]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.size()[0]
        max_len = trg.size()[1]
        #trg_vocab_size = len(vocab)

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        x, (hidden, cell) = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs


class BiDAF(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, device, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.hidden_size = hidden_size

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

        self.dec = layers.Decoder(input_size=hidden_size,
                                  output_dim=88444,
                                  word_vectors=word_vectors,
                                  hidden_size=hidden_size,
                                  n_layers=1,
                                  dropout=drop_prob)

        self.device = device

    def forward(self, sw_idxs, sc_idxs, aw_idxs, ac_idxs, qw_idxs, teacher_forcing_ratio=0.5):
        batch_size = sw_idxs.size()[0]
        max_len = qw_idxs.size()[1]

        s_mask = torch.zeros_like(sw_idxs) != sw_idxs
        a_mask = torch.zeros_like(aw_idxs) != aw_idxs
        s_len, a_len = s_mask.sum(-1), a_mask.sum(-1)

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, 88444).to(self.device)

        s_emb = self.emb(sw_idxs, sc_idxs)   # (batch_size, s_len, hidden_size)
        a_emb = self.emb(aw_idxs, ac_idxs)   # (batch_size, a_len, hidden_size)

        s_enc, (_, s_cell) = self.enc(s_emb, s_len)   # (batch_size, s_len, 2 * hidden_size)
        a_enc, _ = self.enc(a_emb, a_len)   # (batch_size, a_len, 2 * hidden_size)

        att = self.att(s_enc, a_enc, s_mask, a_mask)   # (batch_size, s_len, hidden_size)

        _, (hidden, cell) = self.mod(att, s_len)

        # we add the forward and backward cells of the top LSTM layer
        hidden = torch.add(hidden[-1], hidden[-2]).unsqueeze(0)
        cell = torch.add(cell[-1], cell[-2]).unsqueeze(0)

        dec_input = qw_idxs[:, 0]

        for t in range(1, max_len):
            output, hidden, cell = self.dec(dec_input, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]   # (batch_size)
            dec_input = (qw_idxs[:, t] if teacher_force else top1)

        return outputs
