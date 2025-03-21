import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from preprocess import processed_sequences

###
# TRAINS THE MODEL
###

input_sequences, target_sequences, word_index, max_input_len, max_output_len = processed_sequences # unpacks processed_sequences

X = np.array(input_sequences)
y_input = np.array([seq[:-1] for seq in target_sequences])
y_target = np.array([seq[1:] for seq in target_sequences]) # shifts the target array by 1 for next word prediction

# Set hyperparameters
vocab_size = len(word_index) + 1  # Plus one for padding
embedding_dim = 128
lstm_units = 256

encoder_input = Input(shape=(max_input_len,))
encoder_embed = Embedding(vocab_size, embedding_dim)(encoder_input)
_, state_h, state_c = LSTM(lstm_units, return_state=True)(encoder_embed)

decoder_input = Input(shape=(max_output_len - 1,))
decoder_embed = Embedding(vocab_size, embedding_dim)(decoder_input)
decoder_lstm = LSTM(lstm_units, return_sequences=True)
decoder_output = decoder_lstm(decoder_embed, initial_state=[state_h, state_c])
output = Dense(vocab_size, activation='softmax')(decoder_output)

model = Model([encoder_input, decoder_input], output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([input_sequences, y_input], np.expand_dims(y_target, -1), epochs=50, batch_size=64)

model.save('custom_llm_model.h5')