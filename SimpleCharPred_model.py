'''
Apply a simple RNN to predict various lengths of character sequences
This is just meant to be a fun little exercise

RNN_SimpleCharPrediction.py -> Formats data and saves them to files
SimpleCharPred_model.py -> Trains model using formatted data
'''
from __future__ import print_function
import numpy as np
import psutil
import pickle
import os
np.random.seed(327)

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

X_filename = 'bigtext_X.npy' # file to get X from
y_filename = 'bigtext_y.npy' # file to get y from
window_size = 25

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

int2char_dic = load_obj('int2char')
X = np.load(X_filename)
y = np.load(y_filename)

# Keras model
model = Sequential()
model.add(LSTM(128, input_shape=(window_size, 53)))
model.add(Dense(53))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Training
model.fit(X, y, batch_size=128, nb_epoch=30)

# Saving
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
