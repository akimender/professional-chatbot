import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('custom_llm_model.h5')

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

word_index = tokenizer.word_index
index_word = {i: w for w, i in word_index.items()}

def sample_from_probs(probs, top_k=5):
    top_k_indices = np.argsort(probs)[-top_k:]
    top_k_probs = probs[top_k_indices]
    top_k_probs /= np.sum(top_k_probs)
    return np.random.choice(top_k_indices, p=top_k_probs)

def generate_response(user_input, max_input_len, max_output_len):
    # Tokenize + pad user input
    input_seq = tokenizer.texts_to_sequences([user_input])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')

    # start decoder with <start> token
    start_token = tokenizer.word_index.get('<start>')
    end_token = tokenizer.word_index.get('<end>')

    if start_token is None or end_token is None:
        raise ValueError("Tokenizer is missing <start> or <end> token. Make sure your training targets include them.")

    decoder_input = [start_token]

    response_ids = []

    for _ in range(max_output_len - 1):
        # Pad decoder input
        dec_input_padded = pad_sequences([decoder_input], maxlen=max_output_len - 1, padding='post')
        
        # Predict next token
        prediction = model.predict([input_seq, dec_input_padded], verbose=0)[0]
        next_token_probs = prediction[len(decoder_input) - 1]  # get the timestep we're predicting at
        next_token_id = sample_from_probs(next_token_probs)

        # Break if <end> is predicted
        if next_token_id == end_token or next_token_id == 0:
            break

        response_ids.append(next_token_id)
        decoder_input.append(next_token_id)

    # Convert IDs to words
    response_words = [index_word.get(idx, '') for idx in response_ids]
    return ' '.join(response_words).strip()


max_input_len = model.input[0].shape[1]
max_output_len = model.input[1].shape[1] + 1

# testing manually in console
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    bot_response = generate_response(user_input, max_input_len, max_output_len)
    print(f"Bot: {bot_response}")