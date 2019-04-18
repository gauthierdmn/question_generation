# external libraries
import numpy as np
import pickle
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# internal utilities
import config
from model import BiDAF
from data_loader import SquadDataset
from utils import save_checkpoint, compute_batch_metrics

# preprocessing values used for training
prepro_params = {
    "max_words": config.max_words,
    "word_embedding_size": config.word_embedding_size,
    "char_embedding_size": config.char_embedding_size,
    "max_len_sentence": config.max_len_sentence,
    "max_len_word": config.max_len_word
}

# hyper-parameters setup
hyper_params = {
    "num_epochs": config.num_epochs,
    "batch_size": config.batch_size,
    "learning_rate": config.learning_rate,
    "hidden_size": config.hidden_size,
    "char_channel_width": config.char_channel_width,
    "char_channel_size": config.char_channel_size,
    "drop_prob": config.drop_prob,
    "cuda": config.cuda,
    "pretrained": config.pretrained
}

experiment_params = {"preprocessing": prepro_params, "model": hyper_params}

# train on GPU if CUDA variable is set to True (a GPU with CUDA is needed to do so)
device = torch.device("cuda" if hyper_params["cuda"] else "cpu")
torch.manual_seed(42)

# define a path to save experiment logs
experiment_path = "output/{}".format(config.exp)
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

# save the preprocesisng and model parameters used for this training experiemnt
with open(os.path.join(experiment_path, "config_{}.json".format(config.exp)), "w") as f:
    json.dump(experiment_params, f)

# start TensorboardX writer
writer = SummaryWriter(experiment_path)

# open features file and store them in individual variables (train + dev)
train_features = np.load(os.path.join(config.train_dir, "train_features.npz"))
t_w_sentence, t_c_sentence, t_w_question, t_c_question, t_w_answer, t_c_answer = train_features["sentence_idxs"],\
                                                                                 train_features["sentence_char_idxs"],\
                                                                                 train_features["question_idxs"],\
                                                                                 train_features["question_char_idxs"],\
                                                                                 train_features["answer_idxs"],\
                                                                                 train_features["answer_char_idxs"]

dev_features = np.load(os.path.join(config.dev_dir, "dev_features.npz"))
d_w_context, d_c_context, d_w_question, d_c_question, d_w_answer, d_c_answer = dev_features["sentence_idxs"],\
                                                                               dev_features["sentence_char_idxs"],\
                                                                               dev_features["question_idxs"],\
                                                                               dev_features["question_char_idxs"],\
                                                                               dev_features["answer_idxs"],\
                                                                               dev_features["answer_char_idxs"]

# load the embedding matrix created for our word vocabulary
with open(os.path.join(config.train_dir, "word_embeddings.pkl"), "rb") as e:
    word_embedding_matrix = pickle.load(e)
with open(os.path.join(config.train_dir, "char_embeddings.pkl"), "rb") as e:
    char_embedding_matrix = pickle.load(e)

# load mapping between words and idxs
with open(os.path.join(config.train_dir, "word2idx.pkl"), "rb") as f:
    word2idx = pickle.load(f)

idx2word = dict([(y, x) for x, y in word2idx.items()])
print("Len vocab:", len(idx2word))

# transform them into Tensors
word_embedding_matrix = torch.from_numpy(np.array(word_embedding_matrix)).type(torch.float32)
char_embedding_matrix = torch.from_numpy(np.array(char_embedding_matrix)).type(torch.float32)

# load datasets
train_dataset = SquadDataset(t_w_sentence, t_c_sentence, t_w_question, t_c_question, t_w_answer, t_c_answer)
valid_dataset = SquadDataset(d_w_context, d_c_context, d_w_question, d_c_question, d_w_answer, d_c_answer)

# load data generators
train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=hyper_params["batch_size"],
                              num_workers=4)

valid_dataloader = DataLoader(valid_dataset,
                              shuffle=True,
                              batch_size=hyper_params["batch_size"],
                              num_workers=4)

print("Length of training data loader is:", len(train_dataloader))
print("Length of valid data loader is:", len(valid_dataloader))

# load the model
model = BiDAF(word_vectors=word_embedding_matrix,
              char_vectors=char_embedding_matrix,
              hidden_size=hyper_params["hidden_size"],
              device=device)
if hyper_params["pretrained"]:
    model.load_state_dict(torch.load(os.path.join(experiment_path, "model.pkl"))["state_dict"])
model.to(device)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), hyper_params["learning_rate"], weight_decay=1e-4)

# best loss so far
if hyper_params["pretrained"]:
    best_valid_loss = torch.load(os.path.join(experiment_path, "model.pkl"))["best_valid_loss"]
    epoch_checkpoint = torch.load(os.path.join(experiment_path, "model_last_checkpoint.pkl"))["epoch"]
    print("Best validation loss obtained after {} epochs is: {}".format(epoch_checkpoint, best_valid_loss))
else:
    best_valid_loss = 100
    epoch_checkpoint = 0

# train the Model
print("Starting training...")
for epoch in range(hyper_params["num_epochs"]):
    print("##### epoch {:2d}".format(epoch + 1))
    model.train()
    train_losses = 0
    for i, batch in enumerate(train_dataloader):
        w_sentence, c_sentence, w_question, c_question, w_answer, c_answer = batch[0].long().to(device),\
                                                                             batch[1].long().to(device), \
                                                                             batch[2].long().to(device), \
                                                                             batch[3].long().to(device), \
                                                                             batch[4].long().to(device),\
                                                                             batch[5].long().to(device)
        optimizer.zero_grad()
        pred = model(w_sentence, c_sentence, w_answer, c_answer, w_question)

        pred = pred.view(-1, pred.size(-1))
        w_question = w_question.view(-1)

        loss = criterion(pred, w_question)
        train_losses += loss.item()

        loss.backward()
        optimizer.step()

    writer.add_scalars("train", {"loss": np.round(train_losses / len(train_dataloader), 2),
                                 "epoch": epoch + 1})
    print("Train loss of the model at epoch {} is: {}".format(epoch + 1, np.round(train_losses /
                                                                                  len(train_dataloader), 2)))

    model.eval()
    valid_losses = 0
    valid_em = 0
    valid_f1 = 0
    n_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_dataloader):
            w_sentence, c_sentence, w_question, c_question, w_answer, c_answer = batch[0].long().to(device), \
                                                                                 batch[1].long().to(device), \
                                                                                 batch[2].long().to(device), \
                                                                                 batch[3].long().to(device), \
                                                                                 batch[4].long().to(device), \
                                                                                 batch[5].long().to(device)

            pred = model(w_sentence, c_sentence, w_answer, c_answer, w_question)

            pred = pred.view(-1, pred.size(-1))
            w_question = w_question.view(-1)

            loss = criterion(pred, w_question)
            valid_losses += loss.item()

            n_samples += w_sentence.size(0)

        writer.add_scalars("valid", {"loss": np.round(valid_losses / len(valid_dataloader), 2),
                                     "epoch": epoch + 1})
        print("Valid loss of the model at epoch {} is: {}".format(epoch + 1, np.round(valid_losses /
                                                                                      len(valid_dataloader), 2)))

    # save last model weights
    save_checkpoint({
        "epoch": epoch + 1 + epoch_checkpoint,
        "state_dict": model.state_dict(),
        "best_valid_loss": np.round(valid_losses / len(valid_dataloader), 2)
    }, True, os.path.join(experiment_path, "model_last_checkpoint.pkl"))

    # save model with best validation error
    is_best = bool(np.round(valid_losses / len(valid_dataloader), 2) < best_valid_loss)
    best_valid_loss = min(np.round(valid_losses / len(valid_dataloader), 2), best_valid_loss)
    save_checkpoint({
        "epoch": epoch + 1 + epoch_checkpoint,
        "state_dict": model.state_dict(),
        "best_valid_loss": best_valid_loss
    }, is_best, os.path.join(experiment_path, "model.pkl"))

# export scalar data to JSON for external processing
writer.export_scalars_to_json(os.path.join(experiment_path, "all_scalars.json"))
writer.close()
