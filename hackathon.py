from gensim.models import KeyedVectors
from gensim.test.utils import common_texts
from gensim.models import word2vec
from scipy import spatial
import numpy as np
import gensim.downloader
import os
from semantle_sim import semantle

# Load the model (Uncomment the line you need)
# model = gensim.downloader.load('word2vec-google-news-300')
model_path = './semantle/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

def find_similar_words(guess, semanticscore, words_to_consider):
    arr = {}
    for word in words_to_consider:
        similarity = abs((np.dot(model[guess], model[word])/(np.linalg.norm(model[guess])* np.linalg.norm(model[word])) * 100) - semanticscore)
        if similarity <= 0.01:
            arr[word] = similarity
    return arr

def sort_results_by_value(result):
    return dict(sorted(result.items(), key=lambda item: item[1]))

def square_rooted(x):
    return round(np.sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),4) * 100

words = [w for w in model.key_to_index if w[0].islower() and "_" not in w and "." not in w]

# Example usage
word = "apple"
prev = []
# while(True):
#     score = float(input(f"enter score for {word}:"))
#     print(score)
#     result = find_similar_words(word, score)
#     sorted_result = sort_results_by_value(result).keys()
    
#     if prev==[]:
#         prev = sorted_result
#     prev = list(set(prev).intersection(set(sorted_result)))
#     print(prev)
#     word = prev[0]

semantle_game = semantle(model_path)
current_word = "apple"
print(f"Goal word: {semantle_game.get_goal_word()}\n\n")
print(f"Current word: {current_word}")
similarity = semantle_game.get_similarity(current_word)
print(f"Similarity score for {current_word}: {similarity}%")
possible_words = find_similar_words(current_word, similarity, words)
print(f"Possible words: {possible_words}\n\n")

while similarity != 110 or (len(possible_words) != 0):
    current_word = np.random.choice(list(possible_words.keys()))
    print(f"Current word: {current_word}")
    similarity = semantle_game.get_similarity(current_word)
    print(f"Similarity score for {current_word}: {similarity}%")
    possible_words = find_similar_words(current_word, similarity, possible_words)
    print(f"Possible words: {possible_words}\n\n")