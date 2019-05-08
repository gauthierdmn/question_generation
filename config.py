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
glove = "/Users/gdamien/Data/glove.6B/glove.6B.{}d.txt"
squad_models = "/Users/gdamien/Data/squad/models/"

# preprocessing values
max_words = 40000
word_embedding_size = 100
char_embedding_size = 8
max_len_sentence = 50
max_len_word = 25

# training hyper-parameters
num_epochs = 15
batch_size = 64
learning_rate = 1.0
drop_prob = 0.2
hidden_size = 300
char_channel_width = 5
char_channel_size = 100
cuda = False
pretrained = False

in_vocab_size = 10000
out_vocab_size = 10000
