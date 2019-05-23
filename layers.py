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
        self.attn = Attention(hidden_size=hidden_size, attn_type="general") if attention else None
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
        outputs = torch.zeros(batch_size, self.max_len_sentence - 1, self.output_dim).to(
            self.device) if question is not None else []

        # TODO: we should have a "if bidirectional:" statement here
        if isinstance(enc_hidden, tuple):  # meaning we have a LSTM encoder
            enc_hidden = tuple(
                (torch.cat((hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]), dim=2) for hidden in enc_hidden))
        else:  # GRU layer
            enc_hidden = torch.cat((enc_hidden[0:enc_hidden.size(0):2], enc_hidden[1:enc_hidden.size(0):2]), dim=2)

        enc_out = enc_out[:, -1, :].unsqueeze(1) if not self.attn else enc_out
        dec_hidden = enc_hidden

        if question is not None:  # TRAINING with teacher
            dec_input = question[:, 0].unsqueeze(1)
            input_feed = torch.zeros(batch_size, 1, enc_out.size(2), device=self.device)
            for t in range(0, self.max_len_sentence - 1):
                dec_input = self.embedding(dec_input)  # (batch size, 1, emb dim)
                dec_input = torch.cat((dec_input, input_feed), 2)

                if isinstance(self.rnn, nn.GRU):
                    dec_output, dec_hidden = self.rnn(dec_input, dec_hidden[0])
                else:
                    dec_output, dec_hidden = self.rnn(dec_input, dec_hidden)

                if self.attn:
                    dec_output, p_attn = self.attn(dec_output, enc_out)

                dec_output = self.dropout(dec_output)

                outputs[:, t, :] = self.gen(dec_output)

                dec_input = question[:, t + 1].unsqueeze(1)
                input_feed = dec_output

        else:  # EVALUATION
            dec_input = torch.zeros(enc_out.size(0), 1).fill_(2).long().to(self.device)
            input_feed = torch.zeros(batch_size, 1, enc_out.size(2), device=self.device)
            for t in range(0, self.max_len_sentence):
                dec_input = self.embedding(dec_input)  # (batch size, 1, emb dim)
                dec_input = torch.cat((dec_input, input_feed), 2)

                if isinstance(self.rnn, nn.GRU):
                    dec_output, dec_hidden = self.rnn(dec_input, dec_hidden[0])
                else:
                    dec_output, dec_hidden = self.rnn(dec_input, dec_hidden)

                if self.attn:
                    dec_output, p_attn = self.attn(dec_output, enc_out)

                out = self.gen(dec_output)

                out, probs = sample_sequence(out, self.top_k, self.top_p, self.temperature, self.greedy_decoding)
                if t < self.min_len_sentence and out.item() in self.special_tokens_ids:
                    while out.item() in self.special_tokens_ids:
                        out = torch.multinomial(probs, num_samples=1)

                if out.item() in self.special_tokens_ids:
                    break
                outputs.append(out.item())
                dec_input = out.long().unsqueeze(1)
                input_feed = dec_output

        return outputs


class Generator(nn.Module):
    def __init__(self, decoder_size, output_dim):
        super(Generator, self).__init__()
        self.gen_func = nn.LogSoftmax(dim=-1)
        self.generator = nn.Linear(decoder_size, output_dim)

    def forward(self, x):
        out = self.gen_func(self.generator(x)).squeeze(1)
        return out


class Attention(nn.Module):
    def __init__(self, hidden_size, attn_type="dot"):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type (got {:s}).".format(attn_type))
        self.attn_type = attn_type

        if self.attn_type == "general":
            self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(hidden_size, hidden_size, bias=False)
            self.linear_query = nn.Linear(hidden_size, hidden_size, bias=True)
            self.v = nn.Linear(hidden_size, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size, bias=out_bias)

    def score(self, h_t, h_s):
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            hidden_size = self.hidden_size
            wq = self.linear_query(h_t.view(-1, hidden_size))
            wq = wq.view(tgt_batch, tgt_len, 1, hidden_size)
            wq = wq.expand(tgt_batch, tgt_len, src_len, hidden_size)

            uh = self.linear_context(h_s.contiguous().view(-1, hidden_size))
            uh = uh.view(src_batch, 1, src_len, hidden_size)
            uh = uh.expand(src_batch, tgt_len, src_len, hidden_size)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, hidden_size)).view(tgt_batch, tgt_len, src_len)

    def forward(self, dec_output, enc_output, enc_output_lengths=None):
        batch, source_l, hidden_size = enc_output.size()
        batch_, target_l, hidden_size_ = dec_output.size()

        # compute attention scores, as in Luong et al.
        align = self.score(dec_output, enc_output)

        # Softmax to normalize attention weights
        align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, enc_output)

        # concatenate
        concat_c = torch.cat((c, dec_output), 2).view(batch*target_l, hidden_size*2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, hidden_size)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        attn_h = attn_h.transpose(0, 1).contiguous()
        align_vectors = align_vectors.transpose(0, 1).contiguous()

        return attn_h.permute(1, 0, 2), align_vectors
