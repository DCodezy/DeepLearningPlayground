'''
Apply a simple RNN to predict various lengths of character sequences
This is just meant to be a fun little exercise

RNN_SimpleCharPrediction.py -> Formats data and saves them to files
SimpleCharPred_model.py -> Trains model using formatted data
CharModelUse.py -> Use a trained model to generate new text
'''
from __future__ import print_function
import numpy as np
import psutil
import pickle
import os
np.random.seed(10)

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint

WINDOW_SIZE = 20

with open('data/war_and_peace.txt') as f:
    raw_data = [line for line in f]
    raw_data = ''.join(raw_data)

print('Generating dictionaries...')
raw_data = [c for c in raw_data]
unique_chars = sorted(list(set(raw_data)))
num_unique = len(unique_chars)
char2int_dic = dict((c,i) for (i,c) in enumerate(set(raw_data)))
int2char_dic = dict((i,c) for (i,c) in enumerate(set(raw_data)))
int_data = [char2int_dic[i] for i in raw_data]
print('-' * 20)
print('Details')
print('Total characters: ' + str(len(raw_data)))
print('# unique characters: ' + str(num_unique))
print('Memory used: ' + str(psutil.virtual_memory()[2]) + '%')
print('-' * 20)

print('Creating sequences...')
seq_in = []
exp_out = []
for i in range(0, len(int_data) - WINDOW_SIZE):
    seq_in.append(int_data[i:(i + WINDOW_SIZE)])
    exp_out.append(int_data[i + WINDOW_SIZE])
print('Memory used: ' + str(psutil.virtual_memory()[2]) + '%')

# Vectorize sequences and from to 3d array (n_samples, time_step, input_dim)
# Note: input_dim is to represent the vectorized form of input
print('Creating input vectors...')
X = np.zeros((len(seq_in), WINDOW_SIZE, len(unique_chars)), dtype=np.bool)
y = np.zeros((len(seq_in), len(unique_chars)), dtype=np.bool)
for i, seq in enumerate(seq_in):
    for j, char_as_int in enumerate(seq):
        X[i, j, char_as_int] = 1
        y[i, exp_out[i]] = 1
print('Memory used: ' + str(psutil.virtual_memory()[2]) + '%')


# Keras model
model = Sequential()
model.add(LSTM(128, input_shape=(WINDOW_SIZE, num_unique)))
model.add(Dense(num_unique))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

checkpointer = ModelCheckpoint(filepath='/models/WaP_model1_.{epoch:02d}.hdf5')

# Training
model.fit(X, y, batch_size=128, nb_epoch=30, callbacks=[checkpointer])

# Saving
model_json = model.to_json()
with open("WaP_model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("WaP_model1.h5")
