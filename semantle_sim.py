from gensim.models import KeyedVectors
import numpy as np

class semantle:
    def __init__(self, path_to_model):
        self.model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)

        self.words = [w for w in self.model.key_to_index if w[0].islower() and "_" not in w and "." not in w]

        self.goal_word = np.random.choice(self.words)

    def get_similarity(self, word):
        if word not in self.words:
            return -100
        if word == self.goal_word:
            return 110
        return round(self.model.similarity(self.goal_word, word)*100,2)
    
    def get_goal_word(self):
        return self.goal_word
    
    def reset_goal_word(self):
        self.goal_word = np.random.choice(self.words)
        return self.goal_word