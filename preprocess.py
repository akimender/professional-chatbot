import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# standard downloads for preprocessing, but this will mainly be needed when dealing with documents
# in the case of this project, I'm using pre-cleaned text files
nltk.download('punkt')
nltk.download('stopwords')

file_path = 'data/raw_text.txt'

###
# Returns a string by reading from a txt file
###
def get_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

###
# Returns a string of tokenized words based on a string input
###
def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')

    text = text.encode('ascii', 'ignore').decode('utf-8') # removes non-ascii characters
    text = text.lower() # converts to lowercase
    text = re.sub(r'\d+', '', text) # removes numbers
    text = re.sub(r'\s+', ' ', text) # normalizes spacing

    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    custom_stopwords = set()
    stop_words.update(custom_stopwords)

    tokens = [token for token in tokens if token not in stop_words]
    
    return " ".join(tokens)

def print_sample():
    sample_text = get_text(file_path)
    preprocessed_text = preprocess_text(sample_text)
    print(preprocessed_text)

print_sample()