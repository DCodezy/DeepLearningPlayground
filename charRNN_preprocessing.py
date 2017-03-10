'''
Apply a simple RNN to predict various lengths of character sequences
This is just meant to be a fun little exercise

charRNN_preprocessing.py -> Formats data and saves them to files
charRNN_training.py -> Trains model using formatted data
charRNN_predicting.py -> Use a trained model to generate new text
'''
from __future__ import print_function
import numpy as np
import itertools
import psutil
import pickle
np.random.seed(327)

remove_chars = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '\t',
    '=',
    '@',
    '[',
    ']',
    '(',
    ')',
    '<',
    '>',
    '^',
    '_',
    '|',
    '~',
    '%',
    '*',
    '“',
    '”',
    ',',
    'à',
    'á',
    'â',
    'ä',
    'æ',
    'ç',
    'è',
    'é',
    'ê',
    'ë',
    'í',
    'î',
    'ï',
    'ó',
    'ô',
    'ö',
    'ú',
    'ü',
    'ý',
    'œ',
    '—',
    '‘',
    '’',
    '“',
    '”',
    '\ufeff',
    '"',
    ':',
    ';',
    '/',
    '-'
]

with open('data/war_and_peace.txt', encoding='utf8') as file:
    raw_string = [(line.strip() + ' ') for line in file]
raw_string = ''.join(raw_string)

chars = [c for c in list(raw_string) if c not in remove_chars]

temp_string = ''.join(chars)
temp_string = temp_string.replace('  ', ' ')
temp_string = temp_string.lower()
temp_string = temp_string.replace('!', '.')
temp_string = temp_string.replace('?', '.')

with open('tmp/war_and_peace.txt', 'w') as file:
    file.write(temp_string)


# window_size = 25
'''
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

with open('data/big.txt') as f:
    raw_data = np.array(
        list(
            itertools.chain.from_iterable(
                [list(line.rstrip().strip('\t').lower()) for line in f]
            )
        )
    )

# Remote unwanted chars and translate chars to ints
print("Generating dictionaries...")
raw_data = [c for c in raw_data if c not in remove_chars]
unique_chars = sorted(list(set(raw_data)))
char2int_dic = dict((c,i) for (i,c) in enumerate(set(raw_data)))
int2char_dic = dict((i,c) for (i,c) in enumerate(set(raw_data)))
int_data = [char2int_dic[i] for i in raw_data]

save_obj(char2int_dic, 'char2int')
save_obj(int2char_dic, 'int2char')
print("Saved dictionaries")

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
'''
