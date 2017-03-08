'''
Apply a simple RNN to predict various lengths of character sequences
This is just meant to be a fun little exercise
'''
from __future__ import print_function
import numpy as np
import itertools
np.random.seed(327)  # for reproducibility

# Import data from data_dir, format to X_train, y_trian, etc.
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

# Translate to
unique_chars = sorted(list(set(raw_data)))
raw_data = [c for c in raw_data if c not in remove_chars]
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

# Vectorize sequences and from to 3d array (n_samples, time_step, input_dim)
# Note: input_dim is to represent the vectorized form of input
X = np.zeroes(len(seq_in), window_size, )

# Create and compile Keras net
