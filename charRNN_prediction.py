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
np.random.seed(327)

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

SEED_LENGTH = 25
END_SENTECE_LENGTH  = 400
SENTENCE_SEED = "then there was only one w" # Should be length seed_length

if len(SENTENCE_SEED) != SEED_LENGTH:
    print("Error: incorrected sentecen seed length")
    print(
        "Should be: " + str(SEED_LENGTH)
            + " Actually: " + str(len(SENTENCE_SEED))
    )

print("Loading dictionaries...")
int2char_dic = load_obj('int2char')
char2int_dic = load_obj('char2int')
print("Loaded dictionaries")

print("Loading model")
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Convert input string into usuable sequences
model_seed_input = np.zeros((1, len(SENTENCE_SEED), 53), dtype=np.bool)

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
    model_seed_input[0, SEED_LENGTH - 1] = np.zeros((53), dtype=np.bool)
    model_seed_input[0, SEED_LENGTH - 1, pred_int] = 1

print("Input: ")
print(SENTENCE_SEED)
print("Output: ")
print(output_sentence)
