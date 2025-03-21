import re
import nltk
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# standard downloads for preprocessing, but this will mainly be needed when dealing with documents
# in the case of this project, I'm using pre-cleaned text files
nltk.download('punkt')
nltk.download('stopwords')

file_path = 'data/simple.txt'

###
# Returns a string by reading from a txt file
###
def get_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

text = get_text(file_path)

"""
Returns a string of tokenized words based on a string input

Modifications:
- strip extra whitespace
- tokenize words while keeping punctuation
"""
def preprocess_text(text):
    # define a list of pairs of example user inputs and bot responses
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    pairs = []
    for i in range(0, len(lines), 2): # iterate through lines 2 at a time
        if lines[i].startswith("User:") and lines[i+1].startswith("Bot:"): # ensures that User: and Bot: format is followed
            user = lines[i].replace("User:", "").strip() # removes "User: "
            bot = lines[i+1].replace("Bot:", "").strip() # removes "Bot: "
            pairs.append((user, bot)) # adds the two strings as a tuple
    
    # extracts user inputs and bot responses
    user_inputs = [pair[0] for pair in pairs]
    bot_responses = [pair[1] for pair in pairs]

    # allows for shared vocabulary between inputs and bot responses
    tokenizer = Tokenizer(oov_token="<OOV>", filters='')
    tokenizer.fit_on_texts(user_inputs + ['<start> ' + r + ' <end>' for r in bot_responses])
    word_index = tokenizer.word_index

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # converts the tokens to sequences
    input_sequences = tokenizer.texts_to_sequences(user_inputs)
    target_sequences = tokenizer.texts_to_sequences(['<start> ' + r + ' <end>' for r in bot_responses])

    # pads the sequences by finding the max length of the sequence
    max_input_len = max(len(seq) for seq in input_sequences)
    max_output_len = max(len(seq) for seq in target_sequences)
    input_sequences = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
    target_sequences = pad_sequences(target_sequences, maxlen=max_output_len, padding='post')   

    return input_sequences, target_sequences, word_index, max_input_len, max_output_len

processed_sequences = preprocess_text(text) # sends sequences to model.py as a tuple (input_sequences, target_sequences, word_index, max_input_len, max_output_len)

### HELPER METHODS BELOW ###

def print_sample():
    sample_text = get_text(file_path)
    preprocessed_text = preprocess_text(sample_text)
    print(preprocessed_text)