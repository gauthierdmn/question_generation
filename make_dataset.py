# external libraries
import os
import tqdm
import json
import zipfile
import tarfile
import pickle
import numpy as np
import urllib.request

# internal utilities
import config
from utils import tokenizer, clean_text, word_tokenize, sent_tokenize, build_vocab, build_embeddings, convert_idx

# URL to download SQuAD dataset 2.0
url = "https://rajpurkar.github.io/SQuAD-explorer/dataset"


def maybe_download_squad(url, filename, out_dir):
    # path for local file.
    save_path = os.path.join(out_dir, filename)

    # check if the file already exists
    if not os.path.exists(save_path):
        # check if the output directory exists, otherwise create it.
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print("Downloading", filename, "...")

        # download the dataset
        url = os.path.join(url, filename)
        file_path, _ = urllib.request.urlretrieve(url=url, filename=save_path)

    print("File downloaded successfully!")

    if filename.endswith(".zip"):
        # unpack the zip-file.
        print("Extracting ZIP file...")
        zipfile.ZipFile(file=filename, mode="r").extractall(out_dir)
        print("File extracted successfully!")
    elif filename.endswith((".tar.gz", ".tgz")):
        # unpack the tar-ball.
        print("Extracting TAR file...")
        tarfile.open(name=filename, mode="r:gz").extractall(out_dir)
        print("File extracted successfully!")


class SquadPreprocessor:
    def __init__(self, data_dir, train_filename, dev_filename, tokenizer):
        self.data_dir = data_dir
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.data = None
        self.tokenizer = tokenizer

    def load_data(self, filename="train-v2.0.json"):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath) as f:
            self.data = json.load(f)

    def split_data(self, filename):
        self.load_data(filename)
        sub_dir = filename.split('-')[0]

        # create a subdirectory for Train and Dev data
        if not os.path.exists(os.path.join(self.data_dir, sub_dir)):
            os.makedirs(os.path.join(self.data_dir, sub_dir))

        with open(os.path.join(self.data_dir, sub_dir, sub_dir + '.sentence'), 'w', encoding="utf-8") as sentence_file,\
             open(os.path.join(self.data_dir, sub_dir, sub_dir + '.question'), 'w', encoding="utf-8") as question_file,\
             open(os.path.join(self.data_dir, sub_dir, sub_dir + '.answer'), 'w', encoding="utf-8") as answer_file:

            # loop over the data
            for article_id in tqdm.tqdm(range(len(self.data['data']))):
                list_paragraphs = self.data['data'][article_id]['paragraphs']
                # loop over the paragraphs
                for paragraph in list_paragraphs:
                    context = paragraph['context']
                    context = clean_text(context)
                    context_tokens = word_tokenize(context)
                    context_sentences = sent_tokenize(context)
                    spans = convert_idx(context, context_tokens)
                    num_tokens = 0
                    sent_starts = []
                    for sentence in context_sentences:
                        first_sentence_span = spans[num_tokens][0]
                        num_tokens += len(sentence)
                        sent_starts.append(first_sentence_span)
                    qas = paragraph['qas']
                    # loop over Q/A
                    for qa in qas:
                        question = qa['question']
                        question = clean_text(question)
                        question_tokens = word_tokenize(question)
                        if sub_dir == "train":
                            # select only one ground truth, the top answer, if any answer
                            answer_ids = 1 if qa['answers'] else 0
                        else:
                            answer_ids = len(qa['answers'])
                        if answer_ids:
                            for answer_id in range(answer_ids):
                                answer = qa['answers'][answer_id]['text']
                                answer = clean_text(answer)
                                answer_tokens = word_tokenize(answer)
                                answer_start = qa['answers'][answer_id]['answer_start']
                                sentence_tokens = []
                                for idx, start in enumerate(sent_starts):
                                    if answer_start >= start:
                                        sentence_tokens = context_sentences[idx]
                                    else:
                                        break
                                if not sentence_tokens:
                                    print("Sentence cannot be found")
                                    raise Exception()

                            # write to file
                            sentence_file.write(' '.join([token for token in sentence_tokens]) + '\n')
                            question_file.write(' '.join([token for token in question_tokens]) + '\n')
                            answer_file.write(' '.join([token for token in answer_tokens]) + '\n')

    def preprocess(self):
        self.split_data(train_filename)
        self.split_data(dev_filename)

    def extract_features(self, max_len_sentence=config.max_len_sentence,
                         max_len_word=config.max_len_word, is_train=True):
        # choose the right directory
        directory = "train" if is_train else "dev"

        # load context
        with open(os.path.join(self.data_dir, directory, directory + ".sentence"), "r", encoding="utf-8") as s:
            sentence = s.readlines()
        # load questions
        with open(os.path.join(self.data_dir, directory, directory + ".question"), "r", encoding="utf-8") as q:
            question = q.readlines()
        # load answer
        with open(os.path.join(self.data_dir, directory, directory + ".answer"), "r", encoding="utf-8") as a:
            answer = a.readlines()

        # clean and tokenize context and question
        sentence = [[w for w in word_tokenize(clean_text(doc.strip('\n')))] for doc in sentence]
        question = [[w for w in word_tokenize(clean_text(doc.strip('\n')))] for doc in question]
        answer = [[w for w in word_tokenize(clean_text(doc.strip('\n')))] for doc in answer]

        # download vocabulary if not done yet
        if None: #directory == "train":
            word_vocab, word2idx, char_vocab, char2idx = build_vocab(directory + ".sentence", directory + ".question",
                                                                     "word_vocab.pkl", "word2idx.pkl", "char_vocab.pkl",
                                                                     "char2idx.pkl", is_train=is_train,
                                                                     max_words=config.max_words)
            # create an embedding matrix from the vocabulary with pretrained vectors (GloVe) for words
            build_embeddings(word_vocab, embedding_path=config.glove, output_path="word_embeddings.pkl",
                             vec_size=config.word_embedding_size)
            build_embeddings(char_vocab, embedding_path="", output_path="char_embeddings.pkl",
                             vec_size=config.char_embedding_size)

        else:
            with open(os.path.join(self.data_dir, "train", "word2idx.pkl"), "rb") as wi,\
                 open(os.path.join(self.data_dir, "train", "char2idx.pkl"), "rb") as ci:
                    word2idx = pickle.load(wi)
                    char2idx = pickle.load(ci)

        print("Number of questions before filtering:", len(question))
        filter = [len(s) < (max_len_sentence - 2) and max([len(w) for w in s]) < max_len_word and len(s) > 3
                  and len(q) < (max_len_sentence - 2) and max([len(w) for w in q]) < max_len_word and len(q) > 3
                  for s, q in zip(sentence, question)]
        sentence, question, answer = zip(*[(s, q, a) for s, q, a, f in zip(
                                          sentence, question, answer, filter) if f])
        print("Number of questions after filtering ", len(question))

        # replace the tokenized words with their associated ID in the vocabulary
        sentence_idxs = []
        sentence_char_idxs = []
        question_idxs = []
        question_char_idxs = []
        answer_idxs = []
        answer_char_idxs = []
        for i, (s, q, a) in tqdm.tqdm(enumerate(zip(sentence, question, answer))):
            # create empty numpy arrays
            sentence_idx = np.zeros([max_len_sentence], dtype=np.int32)
            question_idx = np.zeros([max_len_sentence], dtype=np.int32)
            answer_idx = np.zeros([max_len_sentence], dtype=np.int32)
            sentence_char_idx = np.zeros([max_len_sentence, max_len_word], dtype=np.int32)
            question_char_idx = np.zeros([max_len_sentence, max_len_word], dtype=np.int32)
            answer_char_idx = np.zeros([max_len_sentence, max_len_word], dtype=np.int32)

            # replace 0 values with word and char IDs
            for j, word in enumerate(s):
                if word in word2idx:
                    sentence_idx[j] = word2idx[word]
                else:
                    sentence_idx[j] = 1
                for k, char in enumerate(word):
                    if char in char2idx:
                        sentence_char_idx[j, k] = char2idx[char]
                    else:
                        sentence_char_idx[j, k] = 1
            sentence_idxs.append(sentence_idx)
            sentence_char_idxs.append(sentence_char_idx)

            question_idx[0] = word2idx["--SOS--"]
            question_idx[len(q) + 1] = word2idx["--EOS--"]
            question_char_idx[0, 0] = word2idx["--SOS--"]
            question_char_idx[len(q) + 1, 0] = word2idx["--EOS--"]
            for j, word in enumerate(q):
                if word in word2idx:
                    question_idx[j + 1] = word2idx[word]   # j + 1 because the first token is "--SOS--"
                else:
                    question_idx[j + 1] = 1   # 1 stands for the unknown token
                for k, char in enumerate(word):
                    if char in char2idx:
                        question_char_idx[j + 1, k] = char2idx[char]
                    else:
                        question_char_idx[j + 1, k] = 1   # 1 stands for the unknown token
            question_idxs.append(question_idx)
            question_char_idxs.append(question_char_idx)

            for j, word in enumerate(a):
                if word in word2idx:
                    answer_idx[j] = word2idx[word]
                else:
                    answer_idx[j] = 1
                for k, char in enumerate(word):
                    if char in char2idx:
                        answer_char_idx[j, k] = char2idx[char]
                    else:
                        answer_char_idx[j, k] = 1
            answer_idxs.append(answer_idx)
            answer_char_idxs.append(answer_char_idx)

        # save features as numpy arrays
        np.savez(os.path.join(self.data_dir, directory, directory + "_features"),
                 sentence_idxs=np.array(sentence_idxs),
                 sentence_char_idxs=np.array(sentence_char_idxs),
                 question_idxs=np.array(question_idxs),
                 question_char_idxs=np.array(question_char_idxs),
                 answer_idxs=np.array(answer_idxs),
                 answer_char_idxs=np.array(answer_char_idxs))


if __name__ == "__main__":
    train_filename = "train-v2.0.json"
    dev_filename = "dev-v2.0.json"

    maybe_download_squad(url, train_filename, config.data_dir)
    maybe_download_squad(url, dev_filename, config.data_dir)

    p = SquadPreprocessor(config.data_dir, train_filename, dev_filename, tokenizer)
    #p.preprocess()

    p.extract_features(max_len_sentence=config.max_len_sentence,
                       max_len_word=config.max_len_word, is_train=True)
    p.extract_features(max_len_sentence=config.max_len_sentence,
                       max_len_word=config.max_len_word, is_train=False)
