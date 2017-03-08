'''
Apply a simple RNN to predict various lengths of character sequences
This is just meant to be a fun little exercise
'''
from __future__ import print_function
import numpy as np
import itertools
import psutil
np.random.seed(327)  # for reproducibility

X_filename = 'bigtext_X' # file to save X to
y_filename = 'bigtext_y' # file to save y to
remove_chars = '\t=@[]<>^_|~%*'
window_size = 25

with open('data/big.txt') as f:
    raw_data = np.array(
        list(
            itertools.chain.from_iterable(
                [list(line.rstrip().strip('\t').lower()) for line in f]
            )
        )
    )

# Remote unwanted chars and translate chars to ints
raw_data = [c for c in raw_data if c not in remove_chars]
unique_chars = sorted(list(set(raw_data)))
char2int_dic = dict((c,i) for (i,c) in enumerate(set(raw_data)))
int2char_dic = dict((i,c) for (i,c) in enumerate(set(raw_data)))
int_data = [char2int_dic[i] for i in raw_data]

# Create sequences by moving size of $window_size
seq_in = []
exp_out = []
for i in range(0, len(int_data) - window_size):
    seq_in.append(int_data[i:(i + window_size)])
    exp_out.append(int_data[i + window_size])

print("Unique Characters: " + str(len(unique_chars)))
print("Sequence in len: " + str(len(seq_in)))
print(seq_in[0])
print("exp_out len: " + str(len(exp_out)))
print('-' * 15)
print("Memory used: " + str(psutil.virtual_memory()[2]))

# Vectorize sequences and from to 3d array (n_samples, time_step, input_dim)
# Note: input_dim is to represent the vectorized form of input
X = np.zeros((len(seq_in), window_size, len(unique_chars)), dtype=np.bool)
y = np.zeros((len(seq_in), len(unique_chars)), dtype=np.bool)
for i, seq in enumerate(seq_in):
    for j, char_as_int in enumerate(seq):
        X[i, j, char_as_int] = 1
        y[i, exp_out[i]] = 1

print("X: ")
print(X)
print("y: ")
print(y)
print('-' * 15)
print("Memory used: " + str(psutil.virtual_memory()[2]))

# Save arrays
np.save(X_filename, X)
np.save(y_filename, y)
