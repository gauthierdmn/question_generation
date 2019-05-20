# external libraries
import torch
import os

# internal utilities
from preprocessing import DataPreprocessor
from torchtext import data
from model import Seq2Seq
import config

# preprocessing values used for training
prepro_params = {
    "word_embedding_size": 300,
    "max_len_sentence": 50,
    "min_len_sentence": 5
}

# hyper-parameters setup
hyper_params = {
    "batch_size": 1,
    "hidden_size": 600,
    "n_layers": 2,
    "drop_prob": 0.3,
    "cuda": False,
    "pretrained": False,
    "min_len_sentence": 5,
    "max_len_sentence": 50,
    "top_k": 0.,
    "top_p": 0.9,
    "temperature": 0.7,
    "greedy_decoding": False
}

# train on GPU if CUDA variable is set to True (a GPU with CUDA is needed to do so)
device = torch.device("cuda" if hyper_params["cuda"] else "cpu")
torch.manual_seed(42)
experiment_path = "output/{}".format(config.exp)

dp = DataPreprocessor()
_, _, vocabs = dp.load_data(os.path.join(config.out_dir, "train-dataset.pt"),
                            os.path.join(config.out_dir, "dev-dataset.pt"),
                            config.glove)

test_dataset = dp.generate_data(os.path.join(config.out_dir, "test"), "sentence",
                                "question", max_len=prepro_params["max_len_sentence"])

test_dataloader = data.BucketIterator(test_dataset,
                                      batch_size=hyper_params["batch_size"],
                                      sort_key=lambda x: len(x.sentence),
                                      shuffle=False)

# load the model
model = Seq2Seq(in_vocab=vocabs["src_vocab"],
                hidden_size=hyper_params["hidden_size"],
                n_layers=hyper_params["n_layers"],
                trg_vocab=vocabs['trg_vocab'],
                device=device,
                drop_prob=hyper_params["drop_prob"])

if hyper_params["pretrained"]:
    if not hyper_params["cuda"]:
        model.load_state_dict(torch.load(os.path.join(experiment_path, "model.pkl"),
                                         map_location=lambda storage, loc: storage)["state_dict"])
    else:
        model.load_state_dict(torch.load(os.path.join(experiment_path, "model.pkl"))["state_dict"])
model.to(device)

best_valid_loss = 100
epoch_checkpoint = 0

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        sentence, len_sentence, question = batch.src[0], batch.src[1], batch.trg[0]
        pred = model(sentence, len_sentence, None)
        pred = [vocabs["trg_vocab"].itos[i] for i in pred if vocabs["trg_vocab"].itos[i]]
        print(" ".join(pred[1:]))
