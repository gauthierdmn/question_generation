# experiment ID
exp = "expY"

# data directories
data_dir = "/Users/gdamien/Data/squad/"
train_dir = data_dir + "train/"
dev_dir = data_dir + "dev/"

# model paths
spacy_en = "/Users/gdamien/Data/spacy/en_core_web_sm-2.0.0/en_core_web_sm/en_core_web_sm-2.0.0"
glove = "/Users/gdamien/Data/glove.6B/glove.6B.{}d.txt"
squad_models = "/Users/gdamien/Data/squad/models/"

# preprocessing values
max_words = -1
word_embedding_size = 100
char_embedding_size = 8
max_len_sentence = 50
max_len_word = 25

# training hyper-parameters
num_epochs = 15
batch_size = 64
learning_rate = 0.5
drop_prob = 0.2
hidden_size = 100
char_channel_width = 5
char_channel_size = 100
cuda = False
pretrained = False
