'''
Apply a simple RNN to predict various lengths of character sequences
This is just meant to be a fun little exercise
'''
# Base python includes
from __future__ import print_function
import numpy as np
import itertools
np.random.seed(327)  # for reproducibility

# Keras includes
from Keras import sequence


# Import data from data_dir, format to X_train, y_trian, etc.
data_dir = 'data/big.txt'
with open('data/big.txt') as file:
    raw_data = np.array(
        list(itertools.chain.from_iterable(
            [list(line.rstrip()) for line in f]
        ))
    )


# Create and compile Keras net
