'''
Apply a simple RNN to predict various lengths of character sequences
This is just meant to be a fun little exercise

charRNN_preprocessing.py -> Formats data and saves them to files
charRNN_training.py -> Trains model using formatted data
charRNN_prediction.py -> Use a trained model to generate new text
'''
from __future__ import print_function
import numpy as np
import pickle
import psutil
import os
np.random.seed(10)

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

# Given probablity distribution, generate a non-uniform random index
# prob_dist - full 1D array of probabilities
# top_n - number of top probability characters to consider
def generate_char(prob_dist, top_n):
    consider_probs_index = np.argsort(prob_dist)[-top_n:]
    consider_probs = prob_dist[consider_probs_index]
    total_prob = np.sum(consider_probs)
    consider_probs = consider_probs / total_prob
    return np.random.choice(consider_probs_index, p=consider_probs)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

SEED_LENGTH = 20
USE_PROB_OUTPUT = True
N_TOP_RANDOM = 3
END_SENTECE_LENGTH  = 400
SENTENCE_SEED = [
    "then there was only ",
    "thats an interesting",
    "how can you even thi",
    "apple bottom jeans d",
    "for everything that "
]

if len(SENTENCE_SEED) != SEED_LENGTH:
    print("Error: incorrected sentecen seed length")
    print(
        "Should be: " + str(SEED_LENGTH)
            + " Actually: " + str(len(SENTENCE_SEED))
    )

with open('data/war_and_peace.txt') as f:
    raw_data = [line for line in f]
    raw_data = ''.join(raw_data)

print("Recreating dictioaries")
raw_data = [c for c in raw_data]
unique_chars = sorted(list(set(raw_data)))
num_unique = len(unique_chars)
char2int_dic = dict((c,i) for (i,c) in enumerate(set(raw_data)))
int2char_dic = dict((i,c) for (i,c) in enumerate(set(raw_data)))
print('-' * 20)
print('Details')
print('Total characters: ' + str(len(raw_data)))
print('# unique characters: ' + str(num_unique))
print('-' * 20)

print("Loading model...")
'''json_file = open('WaP_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("WaP_model1.h5")
print("Loaded model from disk")'''
loaded_model = load_model('models/WaP_model1.20.hdf5')

# Convert input string into usuable sequences
sentence_ints = []
for sent in SENTENCE_SEED:
    sentence_ints.append([char2int_dic[c] for c in sent])

model_seed_input = np.zeros(
    (len(SENTENCE_SEED), SEED_LENGTH, num_unique), dtype=np.bool
)
for (j, sent) in enumerate(sentence_ints):
    for (i, c) in enumerate(sent):
        model_seed_input[j, i, c] = 1

output_sentence = SENTENCE_SEED

for i in range(0, END_SENTECE_LENGTH):
    pred_logprob = loaded_model.predict(model_seed_input, verbose=0)
    new_input = np.zeros((len(SENTENCE_SEED), num_unique), dtype=np.bool)
    for (j, pred) in enumerate(pred_logprob):
        pred_int = 0
        if USE_PROB_OUTPUT:
            pred_int = generate_car(pred, N_TOP_RANDOM)
        else:
            pred_int = np.argmax(pred)
        new_input[j, pred_int] = 1
        output_sentence[j] += int2char_dic[pred_int]

    # Shift model input to include new character
    model_seed_input = np.roll(model_seed_input, shift=-1, axis=1)
    model_seed_input[:, SEED_LENGTH - 1, :] = new_input

for (i, sent) in enumerate(SENTENCE_SEED):
    print('-' * 25)
    print("Input sentence: " + sent)
    print("Output sentence: " + output_sentence[i])
