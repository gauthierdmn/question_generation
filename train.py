# external libraries
import os
import json
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torchtext import data
from tensorboardX import SummaryWriter

# internal utilities
import config
from preprocessing import DataPreprocessor
from model import Seq2Seq
from utils import dress_for_loss, save_checkpoint, correct_tokens, MetricReporter

# Preprocessing values used for training
prepro_params = {
    "word_embedding_size": config.word_embedding_size,
    "answer_embedding_size": config.answer_embedding_size,
    "max_len_context": config.max_len_context,
    "max_len_question": config.max_len_question,
}

# Hyper-parameters setup
hyper_params = {
    "num_epochs": config.num_epochs,
    "batch_size": config.batch_size,
    "learning_rate": config.learning_rate,
    "hidden_size": config.hidden_size,
    "n_layers": config.n_layers,
    "drop_prob": config.drop_prob,
    "start_decay_epoch": config.start_decay_epoch,
    "decay_rate": config.decay_rate,
    "use_answer": config.use_answer,
    "cuda": config.cuda,
    "pretrained": config.pretrained
}

experiment_params = {"preprocessing": prepro_params, "model": hyper_params}

# Train on GPU if CUDA variable is set to True (a GPU with CUDA is needed to do so)
device = torch.device("cuda" if hyper_params["cuda"] else "cpu")
torch.manual_seed(42)

# Define a path to save experiment logs
experiment_path = "output/{}".format(config.exp)
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

# Save the preprocesisng and model parameters used for this training experiment
with open(os.path.join(experiment_path, "config_{}.json".format(config.exp)), "w") as f:
    json.dump(experiment_params, f)

# Start TensorboardX writer
writer = SummaryWriter(experiment_path)

# Preprocess the data
dp = DataPreprocessor()
train_dataset, valid_dataset, vocabs = dp.load_data(os.path.join(config.out_dir, "train-dataset.pt"),
                                                    os.path.join(config.out_dir, "dev-dataset.pt"),
                                                    config.glove)

# Load the data into datasets of mini-batches
train_dataloader = data.BucketIterator(train_dataset,
                                       batch_size=hyper_params["batch_size"],
                                       sort_key=lambda x: len(x.src),
                                       sort_within_batch=True,
                                       device=device,
                                       shuffle=False)


valid_dataloader = data.BucketIterator(valid_dataset,
                                       batch_size=hyper_params["batch_size"],
                                       sort_key=lambda x: len(x.src),
                                       sort_within_batch=True,
                                       device=device,
                                       shuffle=True)

print("Length of training data loader is:", len(train_dataloader))
print("Length of valid data loader is:", len(valid_dataloader))

# Load the model
model = Seq2Seq(in_vocab=vocabs["src_vocab"],
                hidden_size=hyper_params["hidden_size"],
                n_layers=hyper_params["n_layers"],
                trg_vocab=vocabs['trg_vocab'],
                device=device,
                drop_prob=hyper_params["drop_prob"],
                use_answer=hyper_params["use_answer"])

# Resume training if checkpoint
if hyper_params["pretrained"]:
    model.load_state_dict(torch.load(os.path.join(experiment_path, "model.pkl"))["state_dict"])
model.to(device)

# Define loss and optimizer
padding_idx = vocabs['trg_vocab'].stoi["<PAD>"]
criterion = nn.NLLLoss(ignore_index=padding_idx, reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), hyper_params["learning_rate"])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=list(range(hyper_params["start_decay_epoch"],
                                                                       hyper_params["num_epochs"] + 1)),
                                                 gamma=hyper_params["decay_rate"])

# Create an object to report the different metrics
mc = MetricReporter()

# Get the best loss so far when resuming training
if hyper_params["pretrained"]:
    best_valid_loss = torch.load(os.path.join(experiment_path, "model.pkl"))["best_valid_loss"]
    epoch_checkpoint = torch.load(os.path.join(experiment_path, "model_last_checkpoint.pkl"))["epoch"]
    print("Best validation loss obtained after {} epochs is: {}".format(epoch_checkpoint, best_valid_loss))
else:
    best_valid_loss = 10000  # large number
    epoch_checkpoint = 1

# Train the model
print("Starting training...")
for epoch in range(hyper_params["num_epochs"]):
    print("##### epoch {:2d}".format(epoch))
    model.train()
    mc.train()
    scheduler.step()
    for i, batch in enumerate(train_dataloader):
        # Load a batch of input sentences, sentence lengths, questions and potentially answers
        sentence, len_sentence, question = batch.src[0].to(device), batch.src[1].to(device), batch.trg[0].to(device)
        answer = batch.feat.to(device) if hyper_params["use_answer"] else None
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        # Forward pass to get output/logits
        pred = model(sentence, len_sentence, question, answer)
        # Stack the predictions into a tensor to compute the loss
        pred = dress_for_loss(pred)
        # Calculate Loss: softmax --> negative log likelihood
        loss = criterion(pred.view(-1, pred.size(2)), question[:, 1:].contiguous().view(-1))

        # Update the metrics
        num_non_padding, num_correct = correct_tokens(pred, question, padding_idx)
        mc.update_metrics(loss.item(), num_non_padding, num_correct)

        # Getting gradients w.r.t. parameters
        loss.backward()
        # Truncate the gradients if the norm is greater than a threshold
        clip_grad_norm_(model.parameters(), 5.)
        # Updating parameters
        optimizer.step()

    # Compute the loss, accuracy and perplexity for this epoch and push them to TensorboardX
    mc.report_metrics()
    writer.add_scalars("train", {"loss": mc.list_train_loss[-1],
                                 "accuracy": mc.list_train_accuracy[-1],
                                 "perplexity": mc.list_train_perplexity[-1],
                                 "epoch": mc.epoch})

    model.eval()
    mc.eval()
    with torch.no_grad():
        for i, batch in enumerate(valid_dataloader):
            # Load a batch of input sentence, sentence lengths and questions
            sentence, len_sentence, question = batch.src[0].to(device), batch.src[1].to(device), batch.trg[0].to(device)
            answer = batch.feat.to(device) if hyper_params["use_answer"] else None
            # Forward pass to get output/logits
            pred = model(sentence, len_sentence, question,  answer)
            # Stack the predictions into a tensor to compute the loss
            pred = dress_for_loss(pred)
            # Calculate Loss: softmax --> negative log likelihood
            loss = criterion(pred.view(-1, pred.size(2)), question.view(-1))

            # Update the metrics
            num_non_padding, num_correct = correct_tokens(pred, question, padding_idx)
            mc.update_metrics(loss.item(), num_non_padding, num_correct)

        # Compute the loss, accuracy and perplexity for this epoch and push them to TensorboardX
        mc.report_metrics()
        writer.add_scalars("valid", {"loss": mc.list_valid_loss[-1],
                                     "accuracy": mc.list_valid_accuracy[-1],
                                     "perplexity": mc.list_valid_perplexity[-1],
                                     "epoch": mc.epoch})

    # Save last model weights
    save_checkpoint({
        "epoch": mc.epoch + epoch_checkpoint,
        "state_dict": model.state_dict(),
        "best_valid_loss": mc.list_valid_loss[-1],
    }, True, os.path.join(experiment_path, "model_last_checkpoint.pkl"))

    # Save model weights with best validation error
    is_best = bool(mc.list_valid_loss[-1] < best_valid_loss)
    best_valid_loss = min(mc.list_valid_loss, best_valid_loss)
    save_checkpoint({
        "epoch": mc.epoch + epoch_checkpoint,
        "state_dict": model.state_dict(),
        "best_valid_loss": best_valid_loss
    }, is_best, os.path.join(experiment_path, "model.pkl"))

# Export scalar data to TXT file for external processing and analysis
mc.log_metrics(os.path.join(experiment_path, "train_log.txt"))
