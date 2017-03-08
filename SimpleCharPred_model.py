'''
Apply a simple RNN to predict various lengths of character sequences
This is just meant to be a fun little exercise

RNN_SimpleCharPrediction.py -> Formats data and saves them to files
SimpleCharPred_model.py -> Trains model using formatted data
'''
from __future__ import print_function
import numpy as np
import psutil
np.random.seed(327)

X_filename = 'bigtext_X' # file to get X from
y_filename = 'bigtext_y' # file to get y from
window_size = 25
