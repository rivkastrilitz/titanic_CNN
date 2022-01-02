import data as data
import numpy as np
import tensorflow as tf
#tf.compat.v1.disable_v2_behavior()
import pandas as pd
from collections import Counter
from keras import Sequential
from keras.layers import Dropout
from tensorflow.keras.layers import Dense, Dense, LSTM
from tflearn import activation
# from RNN import optimizer

'''
                1)    Fetch Data
'''


feature_sets_train = pd.read_csv('train.csv')
# TODO: Use both datasets to make the embeddings (vocab_to_int map)
feature_sets_test = pd.read_csv('test.csv')
feature_sets_train_tests = pd.concat([feature_sets_train, feature_sets_test])
feature_sets = feature_sets_train

passengers = [' '.join(map(str, passenger[[2,3,4,5,8,9,10,11]])) for passenger in feature_sets.values]
passengers_test = [' '.join(map(str,passenger[[1,2,3,4,7,8,9,10]])) for passenger in feature_sets_test.values]

survived = [passenger[1] for passenger in feature_sets.values]
feature_sets = passengers
feature_sets_test = passengers_test
labels = survived

'''
                2)    Data preprocessing
'''

#from string import punctuation
#all_text = ''.join([c for c in feature_sets if c not in punctuation])
#feature_sets = all_text.split(',')

passengers = [' '.join(map(str,passenger[[0,1,2,3,4,5,7,8,9,11]])) for passenger in feature_sets_train_tests.values]

all_text = ' '.join(passengers)
words = all_text.split()


'''
                3)    Encoding the words
'''


counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

feature_sets_ints = []
feature_sets_ints_test = []
print(feature_sets[0])
for a in range(5):
    print('test test ',feature_sets[a])



for each in feature_sets:
    # print(each)
    try:
        feature_sets_ints.append([vocab_to_int[word] for word in each.split()])
    except KeyError:
        continue


print(feature_sets_test[0])
for each in feature_sets_test:
    try:
        feature_sets_ints_test.append([vocab_to_int[word] for word in each.split()])
    except KeyError:
        continue

feature_set_lens = Counter([len(x) for x in feature_sets_ints])
print("Zero-length feature_sets: {}".format(feature_set_lens[0]))
print("Maximum feature_set length: {}".format(max(feature_set_lens)))


non_zero_idx = [ii for ii, feature_set in enumerate(feature_sets_ints) if len(feature_set) != 0]
non_zero_idx_test = [ii for ii, feature_set in enumerate(feature_sets_ints_test) if len(feature_set) != 0]

feature_sets_ints = [feature_sets_ints[ii] for ii in non_zero_idx]
feature_sets_ints_test = [feature_sets_ints_test[ii] for ii in non_zero_idx_test]

labels = np.array([labels[ii] for ii in non_zero_idx])

seq_len = 24
features = np.zeros((len(feature_sets_ints), seq_len), dtype=int)
for i, row in enumerate(feature_sets_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]
seq_len = 24
features_test = np.zeros((len(feature_sets_ints_test), seq_len), dtype=int)
for i, row in enumerate(feature_sets_ints_test):
    features_test[i, -len(row):] = np.array(row)[:seq_len]


'''
spliting to train validation and test
'''
split_frac = 0.75
split_idx = int(len(features)*split_frac)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x  = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

print(train_x.shape)
print(test_x.shape)
lstm_size = 518
_learning_rate = 1e-3
_decay = 1e-5



model = Sequential()
model.add(LSTM(units=lstm_size, return_sequences=True, input_shape=(train_x.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=lstm_size, return_sequences=True, input_shape=(test_x.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=lstm_size))
model.add(Dropout(0.2))

model.add(Dense(units=1))

# mean_squared_error
model.summary()
# a = tf.keras.losses.binary.
opt = tf.keras.optimizers.Adam(learning_rate=3e-4, decay=1e-5)

model.compile(loss='hinge', optimizer=opt, metrics=['accuracy'])
model.fit(train_x, train_y,epochs=4, validation_data=(test_x,test_y))




score, acc = model.evaluate(test_x, test_y, batch_size=32)

print("\n\n SUMMARY: batch_size=32")
print('\t\t Test score:', score)
print('\t\t Test accuracy:', acc)
