# external libraries
import os
import json
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torchtext import data
from tensorboardX import SummaryWriter

# internal utilities
import config
from preprocessing import DataPreprocessor
from model import Seq2Seq
from utils import save_checkpoint

# preprocessing values used for training
prepro_params = {
    "word_embedding_size": config.word_embedding_size,
    "max_len_sentence": config.max_len_sentence,
}

# hyper-parameters setup
hyper_params = {
    "num_epochs": config.num_epochs,
    "batch_size": config.batch_size,
    "learning_rate": config.learning_rate,
    "hidden_size": config.hidden_size,
    "n_layers": config.n_layers,
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

dp = DataPreprocessor()
train_dataset, eval_dataset, vocabs = dp.load_data(os.path.join(config.out_dir, "train-dataset.pt"),
                                                   os.path.join(config.out_dir, "dev-dataset.pt"),
                                                   config.glove)

train_dataloader = data.BucketIterator(train_dataset,
                                       batch_size=hyper_params["batch_size"],
                                       sort_key=lambda x: len(x.sentence),
                                       shuffle=True)

eval_dataloader = data.BucketIterator(eval_dataset,
                                      batch_size=hyper_params["batch_size"],
                                      sort_key=lambda x: len(x.sentence),
                                      shuffle=True)

print("Length of training data loader is:", len(train_dataloader))
print("Length of valid data loader is:", len(eval_dataloader))

# load the model
model = Seq2Seq(in_vocab=vocabs["src_vocab"],
                hidden_size=hyper_params["hidden_size"],
                n_layers=hyper_params["n_layers"],
                output_dim=len(vocabs["trg_vocab"].itos),
                device=device,
                drop_prob=hyper_params["drop_prob"])

if hyper_params["pretrained"]:
    model.load_state_dict(torch.load(os.path.join(experiment_path, "model.pkl"))["state_dict"])
model.to(device)

# define loss and optimizer
padding_idx = vocabs['trg_vocab'].stoi["<PAD>"]
criterion = nn.NLLLoss(ignore_index=padding_idx, reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), hyper_params["learning_rate"], momentum=0.9, weight_decay=1e-4)

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
    n_correct = 0
    n_samples = 0
    for i, batch in enumerate(train_dataloader):
        sentence, len_sentence, question = batch.src[0].to(device), batch.src[1].to(device), batch.trg[0].to(device)
        optimizer.zero_grad()
        pred = model(sentence, len_sentence, question)
        pred = pred.view(-1, pred.size(2))
        loss = criterion(pred, question.view(-1))

        if (i + 1) % 500 == 0:
            print("loss:", loss.item())

        pred = pred.max(1)[1]
        non_padding = question.view(-1).ne(padding_idx)
        num_correct = pred.eq(question.view(-1)).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()

        train_losses += loss.item()
        n_samples += num_non_padding
        n_correct += num_correct

        loss.backward()
        clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

    writer.add_scalars("train", {"loss": np.round(train_losses / n_samples, 2),
                                 "accuracy": np.round(100 * (n_correct / n_samples), 2),
                                 "epoch": epoch + 1})
    print("Train loss of the model at epoch {} is: {}".format(epoch + 1, np.round(train_losses /
                                                                                  n_samples, 2)))
    print("Train accuracy of the model at epoch {} is: {}".format(epoch + 1,
                                                                  np.round(100 * (n_correct / n_samples), 2)))

    model.eval()
    valid_losses = 0
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            sentence, len_sentence, question = batch.src[0].to(device), batch.src[1].to(device), batch.trg[0].to(device)

            pred = model(sentence, len_sentence, question)

            pred = pred.view(-1, pred.size(2))

            loss = criterion(pred, question.view(-1))

            pred = pred.max(1)[1]
            non_padding = question.view(-1).ne(padding_idx)
            num_correct = pred.eq(question.view(-1)).masked_select(non_padding).sum().item()
            num_non_padding = non_padding.sum().item()

            valid_losses += loss.item()
            n_samples += num_non_padding
            n_correct += num_correct

        writer.add_scalars("valid", {"loss": np.round(valid_losses / n_samples, 2),
                                     "accuracy": np.round(100 * (n_correct / n_samples), 2),
                                     "epoch": epoch + 1})
        print("Valid loss of the model at epoch {} is: {}".format(epoch + 1, np.round(valid_losses /
                                                                                      n_samples, 2)))
        print("Valid accuracy of the model at epoch {} is: {}".format(epoch + 1,
                                                                      np.round(100 * (n_correct / n_samples), 2)))

    # save last model weights
    save_checkpoint({
        "epoch": epoch + 1 + epoch_checkpoint,
        "state_dict": model.state_dict(),
        "best_valid_loss": np.round(valid_losses / len(eval_dataloader), 2)
    }, True, os.path.join(experiment_path, "model_last_checkpoint.pkl"))

    # save model with best validation error
    is_best = bool(np.round(valid_losses / len(eval_dataloader), 2) < best_valid_loss)
    best_valid_loss = min(np.round(valid_losses / len(eval_dataloader), 2), best_valid_loss)
    save_checkpoint({
        "epoch": epoch + 1 + epoch_checkpoint,
        "state_dict": model.state_dict(),
        "best_valid_loss": best_valid_loss
    }, is_best, os.path.join(experiment_path, "model.pkl"))

# export scalar data to JSON for external processing
writer.export_scalars_to_json(os.path.join(experiment_path, "all_scalars.json"))
writer.close()
