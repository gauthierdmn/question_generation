import torch.nn as nn

import layers


class Seq2Seq(nn.Module):
    def __init__(self, in_vocab, hidden_size, output_dim, device, drop_prob=0.):
        super(Seq2Seq, self).__init__()

        self.enc = layers.Encoder(input_size=in_vocab.vectors.size(1),
                                  hidden_size=hidden_size,
                                  num_layers=1,
                                  word_vectors=in_vocab.vectors,
                                  bidirectional=True,
                                  drop_prob=drop_prob)

        self.dec = layers.Decoder(input_size=2 * hidden_size,
                                  hidden_size=hidden_size,
                                  output_dim=output_dim,
                                  word_vectors=in_vocab.vectors,
                                  n_layers=1,
                                  device=device,
                                  dropout=drop_prob,
                                  attention=True)

    def forward(self, sentence, sentence_len, question):
        enc_output, enc_hidden = self.enc(sentence, sentence_len)
        outputs = self.dec(enc_output, enc_hidden, question)

        return outputs
