import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.utils import np_utils
from tensorflow import math
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from helper_functions import encode_list, confusion_matrix, accuracy

# fix random seed for reproducibility
np.random.seed(7)

# parameters
seq_length = 1273
number_proteins = 20
voc_size = number_proteins + 1
number_clades = 17
embedding_dim = 32   # kinda random

# load training and test data
#training_data = pd.read_csv('training_data_mini_sample.csv')
#test_data = pd.read_csv('test_data_mini_sample.csv')
training_data = pd.read_csv('training_data.csv')
test_data = pd.read_csv('test_data.csv')


X_test = encode_list(test_data['Sequence'])
y_test = np.asarray(test_data['Clade'])
y_test_oh = np_utils.to_categorical(y_test, num_classes=number_clades+1)            # for categorical crossentropy


X_train = encode_list(training_data['Sequence'])
y_train = np.asarray(training_data['Clade'])

#print((X_train[0]))
#print((X_train[0]).shape)
#print(y_train_oh[0])
#print((y_train_oh[0]).shape)


# trying to balance dataset
# distribution training data:
# [1: 1328, 2: 1005, 3: 63252, 4: 12641, 5: 4623, 6: 912, 7: 22076, 8: 88, 9: 7460, 10: 22, 11: 35, 12: 2021,
# 13: 198, 14: 4691, 15: 2, 16: 4552, 17: 1]

over_sampling_dict = {6: 1000, 8: 500, 10: 500, 11: 500, 13: 500, 15: 500, 17: 500}
under_sampling_dict = {3: 8000, 4: 4000, 5: 1200,  7: 8000, 9: 2000, 12: 700, 14: 1300, 16: 1300}

over_sampler = RandomOverSampler(sampling_strategy=over_sampling_dict)
under_sampler = RandomUnderSampler(sampling_strategy=under_sampling_dict)

X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)
X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train_resampled, y_train_resampled)
y_train_res_oh = np_utils.to_categorical(y_train_resampled, num_classes=number_clades+1)


# create the model
model = Sequential()
model.add(Embedding(input_dim=voc_size, output_dim=embedding_dim, input_length=seq_length))
model.add(LSTM(100))
model.add(Dense(number_clades+1, activation='sigmoid'))  # use softmax?
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train_resampled, y_train_res_oh, validation_data=(X_test, y_test_oh), epochs=5, batch_size=16)

# Final evaluation of the model
#scores = model.evaluate(X_test, y_test_oh, verbose=0)
#print("Accuracy: %.10f%%" % (scores[1] * 100))

y_pred = model.predict(X_test)
y_pred_dec = math.argmax(y_pred, axis=-1)


print("Accuracy: %.10f%%" % (accuracy(y_test, y_pred_dec)*100))
print(math.confusion_matrix(y_test, y_pred_dec))
for i in range(0, number_clades+1):
    print(confusion_matrix(y_test, y_pred_dec)[i])


# work on balancing the batches
