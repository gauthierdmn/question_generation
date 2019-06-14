# external libraries
import numpy as np
import math
import torch
import torch.nn.functional as F
from spacy.lang.en import English

# internal utilities
import config

tokenizer = English()
tokenizer.add_pipe(tokenizer.create_pipe("sentencizer"))
device = torch.device("cuda" if config.cuda else "cpu")


def clean_text(text):
    text = text.replace("]", " ] ")
    text = text.replace("[", " [ ")
    text = text.replace("\n", " ")
    text = text.replace("''", '" ').replace("``", '" ')

    return text


def word_tokenize(text):
    tokens = [token.text for token in tokenizer(text) if token.text]
    tokens = [t for t in tokens if t.strip("\n").strip()]
    return tokens


def sent_tokenize(text):
    return [[token.text for token in sentence if token.text] for sentence in tokenizer(text).sents]


def feature_tokenize(text, f_sep=u"￨"):
    return [t.split(f_sep)[0] for t in text.split()], [t.split(f_sep)[1] for t in text.split()]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)

    return spans


def dress_for_loss(prediction):
    prediction = torch.stack(prediction).squeeze(0).transpose(0, 1).contiguous()
    return prediction


def correct_tokens(pred, true_tokens, padding_idx):
    pred = pred.view(-1, pred.size(2))
    pred = pred.max(1)[1]
    true_tokens = true_tokens[:, 1:].contiguous()
    non_padding = true_tokens.view(-1).ne(padding_idx)
    num_correct = pred.eq(true_tokens.view(-1)).masked_select(non_padding).sum().item()
    num_non_padding = non_padding.sum().item()
    return num_non_padding, num_correct


def save_checkpoint(state, is_best, filename="/output/checkpoint.pkl"):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best model.")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation loss did not improve.")


class MetricReporter:
    def __init__(self, last_epoch=0, verbose=True):
        self.epoch = last_epoch
        self.verbose = verbose
        self.training = True
        self.losses = 0
        self.n_samples = 0
        self.n_correct = 0
        self.list_train_loss = []
        self.list_train_accuracy = []
        self.list_train_perplexity = []
        self.list_valid_loss = []
        self.list_valid_accuracy = []
        self.list_valid_perplexity = []

    def train(self):
        self.epoch += 1
        self.training = True
        self.clear_metrics()

    def eval(self):
        self.training = False
        self.clear_metrics()

    def update_metrics(self, l, n_s, n_c):
        self.losses += l
        self.n_samples += n_s
        self.n_correct += n_c

    def compute_loss(self):
        return np.round(self.losses / self.n_samples, 2)

    def compute_accuracy(self):
        return np.round(100 * (self.n_correct / self.n_samples), 2)

    def compute_perplexity(self):
        return np.round(math.exp(self.losses / float(self.n_samples)), 2)

    def report_metrics(self):
        # Compute metrics
        set_name = "Train" if self.training else "Valid"
        loss = self.compute_loss()
        accuracy = self.compute_accuracy()
        perplexity = self.compute_perplexity()
        # Print the metrics to std output if verbose is True
        if self.verbose:
            print("{} loss of the model at epoch {} is: {}".format(set_name, self.epoch, loss))
            print("{} accuracy of the model at epoch {} is: {}".format(set_name, self.epoch, accuracy))

            print("{} perplexity of the model at epoch {} is: {}".format(set_name, self.epoch, perplexity))
        # Store the metrics in lists
        if self.train:
            self.list_train_loss.append(loss)
            self.list_train_accuracy.append(accuracy)
            self.list_train_perplexity.append(perplexity)
        else:
            self.list_valid_loss.append(loss)
            self.list_valid_accuracy.append(accuracy)
            self.list_valid_perplexity.append(perplexity)

    def clear_metrics(self):
        self.losses = 0
        self.n_samples = 0
        self.n_correct = 0

    def log_metrics(self, log_filename):
        with open(log_filename, "w") as f:
            f.write("Epochs:" + str(list(range(len(self.list_train_loss)))) + "\n")
            f.write("Train loss:" + str(self.list_train_loss) + "\n")
            f.write("Train accuracy:" + str(self.list_train_accuracy) + "\n")
            f.write("Train perplexity:" + str(self.list_train_perplexity) + "\n")
            f.write("Valid loss:" + str(self.list_valid_loss) + "\n")
            f.write("Valid accuracy:" + str(self.list_valid_accuracy) + "\n")
            f.write("Valid perplexity:" + str(self.list_valid_perplexity) + "\n")


class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

        # for checking if the queue is empty

    def isEmpty(self):
        return len(self.queue) == []

        # for inserting an element in the queue

    def put(self, data):
        self.queue.append(data)

        # for popping an element based on Priority

    def get(self):
        try:
            max = 0
            for i in range(len(self.queue)):
                if self.queue[i][0] > self.queue[max][0]:
                    max = i
            item = self.queue[max]
            del self.queue[max]
            return item
        except IndexError:
            print()
            exit()


class BeamSearchNode:
    def __init__(self, hidden, prevnode, wordid, log_prob, length, inputfeed):
        self.hidden = hidden
        self.prevnode = prevnode
        self.wordid = wordid
        self.logp = log_prob
        self.leng = length
        self.feed = inputfeed

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


class Beam:
    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def put(self, data):
        self.queue.append(data)

    def get(self):
        try:
            max = 0
            for i in range(len(self.queue)):
                if self.queue[i][0] > self.queue[max][0]:
                    max = i
            item = self.queue[max]
            del self.queue[max]
            return item
        except IndexError:
            print()
            exit()


# The below functions are modified versions of functions from:
# https://github.com/huggingface/transfer-learning-conv-ai/blob/master/train.py (MIT License)
def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(logits, top_k, top_p, temperature, greedy_decoding=False):
    logits = logits[0] / temperature
    logits = top_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    pred = torch.topk(probs, 1)[1] if greedy_decoding else torch.multinomial(probs, 1)

    return pred, probs
