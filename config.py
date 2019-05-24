# experiment ID
exp = "qg-1"

# data directories
newsqa_data_dir = "/Users/gdamien/Data/newsqa/newsqa-data-v1"
squad_data_dir = "/Users/gdamien/Data/squad/"
out_dir = "/Users/gdamien/Data/qg/"
train_dir = squad_data_dir + "train/"
dev_dir = squad_data_dir + "dev/"

# model paths
spacy_en = "/Users/gdamien/Data/spacy/en_core_web_sm-2.0.0/en_core_web_sm/en_core_web_sm-2.0.0"
glove = "/Users/gdamien/Data/glove.6B/"
squad_models = "/Users/gdamien/Data/squad/models/"

# preprocessing values
word_embedding_size = 300
max_len_sentence = 50
in_vocab_size = 45000
out_vocab_size = 28000

# training hyper-parameters
num_epochs = 15
batch_size = 64
learning_rate = 1.0
hidden_size = 600
n_layers = 2
drop_prob = 0.3
start_decay_epoch = 7
decay_rate = 0.5
cuda = True
pretrained = False

# eval hyper-parameters
eval_batch_size = 1
min_len_sentence = 5
greedy_decoding = False
top_k = 0.
top_p = 0.9
temperature = 0.7
