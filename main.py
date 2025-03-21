import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('custom_llm_model.h5')

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

word_index = tokenizer.word_index
index_word = {i: w for w, i in word_index.items()}

def generate_response(user_input, max_input_len, max_output_len):
    # Tokenize + pad user input
    input_seq = tokenizer.texts_to_sequences([user_input])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')

    # Create empty decoder input (start from 0s)
    decoder_input = np.zeros((1, max_output_len - 1), dtype=np.int32)

    # Run the model to get prediction
    prediction = model.predict([input_seq, decoder_input])[0]

    # Greedy decode
    predicted_ids = np.argmax(prediction, axis=-1)

    # Convert IDs to words
    response_words = [index_word.get(idx, '') for idx in predicted_ids if idx != 0]
    return ' '.join(response_words).strip()

max_input_len = model.input[0].shape[1]
max_output_len = model.input[1].shape[1] + 1

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    bot_response = generate_response(user_input, max_input_len, max_output_len)
    print(f"Bot: {bot_response}")