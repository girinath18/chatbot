import io
import random
import string  # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK packages (if not already installed)
nltk.download('popular', quiet=True)

# Define functions and constants

# Initialize lemmatizer
lemmer = WordNetLemmatizer()

# Define greeting inputs and responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def LemTokens(tokens):
    """Lemmatize tokens."""
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    """Remove punctuation and normalize text."""
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def greeting(sentence):
    """Return a random greeting response if a greeting is detected in the user's input."""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

def response(user_response, sent_tokens):
    """Generate a chatbot response based on cosine similarity."""
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
    else:
        robo_response = robo_response + sent_tokens[idx]
        
    sent_tokens.pop()  # Remove user input after processing
    return robo_response

def chatbot():
    flag = True
    print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")

    # Read and preprocess the corpus (india.txt)
    with open('dhoni.txt', 'r', errors='ignore') as f:
        raw = f.read().lower()

    sent_tokens = nltk.sent_tokenize(raw)
    word_tokens = nltk.word_tokenize(raw)

    while flag:
        user_response = input().lower()
        if user_response != 'bye':
            if user_response == 'thanks' or user_response == 'thank you':
                flag = False
                print("ROBO: You are welcome..")
            else:
                if greeting(user_response) is not None:
                    print("ROBO: " + greeting(user_response))
                else:
                    print("ROBO: " + response(user_response, sent_tokens))
        else:
            flag = False
            print("ROBO: Bye! take care..")

# Entry point for script
if __name__ == "__main__":
    chatbot()
