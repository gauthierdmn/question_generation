import torch.nn as nn

import layers
import config


class Seq2Seq(nn.Module):
    def __init__(self, in_vocab, hidden_size, n_layers, trg_vocab, device, drop_prob=0., use_answer=True):
        super(Seq2Seq, self).__init__()

        self.enc = layers.Encoder(input_size=in_vocab.vectors.size(1) if not use_answer else in_vocab.vectors.size(1) +
                                  config.answer_embedding_size,
                                  hidden_size=hidden_size,
                                  num_layers=n_layers,
                                  word_vectors=in_vocab.vectors,
                                  bidirectional=True,
                                  drop_prob=drop_prob if n_layers > 1 else 0.)

        self.dec = layers.Decoder(input_size=in_vocab.vectors.size(1) + hidden_size,
                                  hidden_size=hidden_size,
                                  word_vectors=in_vocab.vectors,
                                  trg_vocab=trg_vocab,
                                  n_layers=n_layers,
                                  device=device,
                                  dropout=drop_prob if n_layers > 1 else 0.,
                                  attention=True)

    def forward(self, sentence, sentence_len, question=None, answer=None):
        enc_output, enc_hidden = self.enc(sentence, sentence_len, answer)
        outputs = self.dec(enc_output, enc_hidden, question)

        return outputs
