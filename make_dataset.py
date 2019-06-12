# external libraries
import os
import tqdm
import json
import zipfile
import tarfile
import urllib.request

# internal utilities
import config
from utils import tokenizer, clean_text, word_tokenize, sent_tokenize, convert_idx

# URL to download SQuAD dataset 2.0
squad_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset"


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

        with open(os.path.join(self.data_dir, sub_dir, sub_dir + '.context'), 'w', encoding="utf-8") as context_file,\
             open(os.path.join(self.data_dir, sub_dir, sub_dir + '.sentence'), 'w', encoding="utf-8") as sentence_file,\
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
                    if config.paragraph and (len(context_tokens) < config.min_len_context or len(context_tokens) > config.max_len_context):
                        continue
                    context_sentences = sent_tokenize(context)
                    spans = convert_idx(context, context_tokens)
                    num_tokens = 0
                    first_token_sentence = []
                    for sentence in context_sentences:
                        first_token_sentence.append(num_tokens)
                        num_tokens += len(sentence)
                    qas = paragraph['qas']
                    # loop over Q/A
                    for qa in qas:
                        question = qa['question']
                        question = clean_text(question)
                        question_tokens = word_tokenize(question)
                        if question_tokens[-1] != "?" or len(question_tokens) < config.min_len_question or len(question_tokens) > config.max_len_question:
                            continue
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
                                answer_stop = answer_start + len(answer)

                                # Getting spans of the answer in the context
                                answer_span = []
                                for idx, span in enumerate(spans):
                                    if not (answer_stop <= span[0] or answer_start >= span[1]):
                                        answer_span.append(idx)
                                if not answer_span:
                                    continue

                                # Getting the sentence where we have the answer
                                sentence_tokens = []
                                for idx, start in enumerate(first_token_sentence):
                                    if answer_span[0] >= start:
                                        sentence_tokens = context_sentences[idx]
                                        answer_sentence_span = [span - start for span in answer_span]
                                    else:
                                        break
                                if not sentence_tokens:
                                    print("Sentence cannot be found")
                                    raise Exception()

                            # write to file
                            context_file.write(" ".join([token + u"￨" + "1" if idx in answer_span else token + u"￨" + "0" for idx, token in enumerate(context_tokens)]) + "\n")
                            sentence_file.write(" ".join([token + u"￨" + "1" if idx in answer_sentence_span else token + u"￨" + "0" for idx, token in enumerate(sentence_tokens)]) + "\n")
                            question_file.write(" ".join([token for token in question_tokens]) + "\n")
                            answer_file.write(" ".join([token for token in answer_tokens]) + "\n")

    def preprocess(self):
        self.split_data(self.train_filename)
        self.split_data(self.dev_filename)


class NewsQAPreprocessor:
    def __init__(self, data_dir, filename, tokenizer):
        self.data_dir = data_dir
        self.filename = filename
        self.data = None
        self.tokenizer = tokenizer

    def load_data(self, filename="combined-newsqa-data-v1.json"):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath) as f:
            self.data = json.load(f)

    def split_data(self, filename):
        self.load_data(filename)

        envs = ["train", "dev"]
        for sub_dir in envs:
            # create a subdirectory for Train and Dev data
            if not os.path.exists(os.path.join(self.data_dir, sub_dir)):
                os.makedirs(os.path.join(self.data_dir, sub_dir))

            with open(os.path.join(self.data_dir, sub_dir, sub_dir + ".context"), "w", encoding="utf-8") as context_file,\
                 open(os.path.join(self.data_dir, sub_dir, sub_dir + ".sentence"), "w", encoding="utf-8") as sentence_file,\
                 open(os.path.join(self.data_dir, sub_dir, sub_dir + ".question"), "w", encoding="utf-8") as question_file,\
                 open(os.path.join(self.data_dir, sub_dir, sub_dir + ".answer"), "w", encoding="utf-8") as answer_file:

                # loop over the data
                for article in tqdm.tqdm(self.data["data"]):
                    context = article["text"]
                    context_tokens = word_tokenize(context)
                    context_sentences = sent_tokenize(context)
                    if config.paragraph and (len(context_tokens) < config.min_len_context or len(
                            context_tokens) > config.max_len_context):
                        continue
                    spans = convert_idx(context, context_tokens)
                    num_tokens = 0
                    first_token_sentence = []
                    for sentence in context_sentences:
                        first_token_sentence.append(num_tokens)
                        num_tokens += len(sentence)
                    if not article["type"] == sub_dir:
                        continue
                    for question in article["questions"]:
                        if question.get("isQuestionBad") == 0 and question["consensus"].get("s"):
                            q = question["q"].strip()
                            if q[-1] != "?" or len(q.split()) < config.min_len_question or len(q.split()) > config.max_len_question:
                                continue
                            answer_start = question["consensus"]["s"]
                            answer = context[question["consensus"]["s"]: question["consensus"]["e"]].strip(".| ").strip("\n")
                            answer_stop = answer_start + len(answer)

                            # Getting spans of the answer in the context
                            answer_span = []
                            for idx, span in enumerate(spans):
                                if not (answer_stop <= span[0] or answer_start >= span[1]):
                                    answer_span.append(idx)
                            if not answer_span:
                                continue

                            # Getting the sentence where we have the answer
                            sentence_tokens = []
                            for idx, start in enumerate(first_token_sentence):
                                if answer_span[0] >= start:
                                    sentence_tokens = context_sentences[idx]
                                    answer_sentence_span = [span - start for span in answer_span]
                                else:
                                    break

                            # write to file
                            sent = []
                            for idx, token in enumerate(sentence_tokens):
                                if token.strip("\n").strip():
                                    if idx in answer_sentence_span:
                                        sent.append(token + u"￨" + "1")
                                    else:
                                        sent.append(token + u"￨" + "0")
                            sent = " ".join(sent)
                            sent = sent.strip()
                            index = sent.find("(￨0 CNN￨0 )￨0 --￨0 ")
                            if index > -1:
                                sent = sent[index + len("(￨0 CNN￨0 )￨0 --￨0 "):]

                            ctxt = []
                            for idx, token in enumerate(context_tokens):
                                if token.strip("\n").strip():
                                    if idx in answer_span:
                                        ctxt.append(token + u"￨" + "1")
                                    else:
                                        ctxt.append(token + u"￨" + "0")
                            ctxt = " ".join(ctxt)
                            ctxt = ctxt.strip()
                            index = ctxt.find("(￨0 CNN￨0 )￨0 --￨0 ")
                            if index > -1:
                                ctxt = ctxt[index + len("(￨0 CNN￨0 )￨0 --￨0 "):]

                            context_file.write(ctxt + "\n")
                            sentence_file.write(sent + "\n")
                            question_file.write(q + "\n")
                            answer_file.write(answer + "\n")

    def preprocess(self):
        self.split_data(self.filename)


def concatenate_data(squad_data_dir, newsqa_data_dir, out_dir, env="train", full_context=False):
    ext = ".context" if full_context else ".sentence"
    sentence_files = [os.path.join(squad_data_dir, env, env + ext),
                      os.path.join(newsqa_data_dir, env, env + ext)]
    question_files = [os.path.join(squad_data_dir, env, env + ".question"),
                      os.path.join(newsqa_data_dir, env, env + ".question")]
    out_sentence_filename = os.path.join(out_dir, env + ext)
    out_question_filename = os.path.join(out_dir, env + ".question")

    for infiles, outfile in zip([sentence_files, question_files], [out_sentence_filename, out_question_filename]):
        with open(outfile, "w") as o:
            for f in infiles:
                with open(f) as infile:
                    for line in infile:
                        o.write(line)

    with open(out_sentence_filename, "r") as f,\
         open(out_question_filename, "r") as g:
        sentence_lines = f.readlines()
        question_lines = g.readlines()

    sentence_lines, question_lines = zip(
        *[(s, q) for s, q in sorted(zip(sentence_lines, question_lines), key=lambda x: len(word_tokenize(x[0])))])

    with open(out_sentence_filename, "w") as f,\
         open(out_question_filename, "w") as g:
        for line in sentence_lines:
            f.write(line)
        for line in question_lines:
            g.write(line)


if __name__ == "__main__":
    squad_train_filename = "train-v2.0.json"
    squad_dev_filename = "dev-v2.0.json"
    newsqa_filename = "combined-newsqa-data-v1.json"

    maybe_download_squad(squad_url, squad_train_filename, config.squad_data_dir)
    maybe_download_squad(squad_url, squad_dev_filename, config.squad_data_dir)

    p1 = NewsQAPreprocessor(config.newsqa_data_dir, newsqa_filename, tokenizer)
    p1.preprocess()

    p2 = SquadPreprocessor(config.squad_data_dir, squad_train_filename, squad_dev_filename, tokenizer)
    p2.preprocess()

    concatenate_data(config.squad_data_dir, config.newsqa_data_dir, config.out_dir, env="train", full_context=config.paragraph)
    concatenate_data(config.squad_data_dir, config.newsqa_data_dir, config.out_dir, env="dev", full_context=config.paragraph)
