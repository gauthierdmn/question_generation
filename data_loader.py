# external libraries
import torch.utils.data as data


class SquadDataset(data.Dataset):
    """Custom Dataset for SQuAD data compatible with torch.utils.data.DataLoader."""

    def __init__(self, w_sentence, c_sentence, w_question, c_question, w_answer, c_answer):
        """Set the path for audio data, together wth labels and objid."""
        self.w_sentence = w_sentence
        self.c_sentence = c_sentence
        self.w_question = w_question
        self.c_question = c_question
        self.w_answer = w_answer
        self.c_answer = c_answer

    def __getitem__(self, index):
        """Returns one data trio (context, question, answer)."""
        return self.w_sentence[index], self.c_sentence[index], self.w_question[index], self.c_question[index],\
               self.w_answer[index], self.c_answer[index]

    def __len__(self):
        return len(self.w_sentence)
