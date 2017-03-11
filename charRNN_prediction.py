'''
Apply a simple RNN to predict various lengths of character sequences
This is just meant to be a fun little exercise

RNN_SimpleCharPrediction.py -> Formats data and saves them to files
SimpleCharPred_model.py -> Trains model using formatted data
CharModelUse.py -> Use a trained model to generate new text
'''
from __future__ import print_function
import numpy as np
import pickle
import psutil
import os
np.random.seed(10)

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

SEED_LENGTH = 20
END_SENTECE_LENGTH  = 400
SENTENCE_SEED = "there once was a " # Should be length seed_length

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

print("Loading model")
json_file = open('WaP_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("WaP_model1.h5")
print("Loaded model from disk")

# Convert input string into usuable sequences
model_seed_input = np.zeros((1, len(SENTENCE_SEED), num_unique), dtype=np.bool)

remove_chars = '\t=@[]<>^_|~%*'
mod_sentence_seed = list(SENTENCE_SEED.lower())
sentence_ints = [char2int_dic[c] for c in mod_sentence_seed
						if c not in remove_chars]
for (i, c) in enumerate(sentence_ints):
    model_seed_input[0, i, c] = 1

output_sentence = SENTENCE_SEED

for i in range(0, END_SENTECE_LENGTH):
    pred_logprob = loaded_model.predict(model_seed_input, verbose=0)[0]
    pred_int = np.argmax(pred_logprob)

    output_sentence += int2char_dic[pred_int]

    # Shift model input to include new character
    model_seed_input = np.roll(model_seed_input, shift=-1, axis=1)
    model_seed_input[0, SEED_LENGTH - 1] = np.zeros((num_unique), dtype=np.bool)
    model_seed_input[0, SEED_LENGTH - 1, pred_int] = 1

print("Input: ")
print(SENTENCE_SEED)
print("Output: ")
print(output_sentence)
