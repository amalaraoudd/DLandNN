import nltk
import streamlit as st
import speech_recognition as sr
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the text file and preprocess the data
with open('C:\\Users\\uber\\chat_file.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

# Tokenize the text into sentences
sentences = sent_tokenize(data)

# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence

# Define a function to transcribe speech into text using the speech recognition algorithm
def transcribe_speech():
    # Initialize recognizer class
    r = sr.Recognizer()
    # Reading Microphone as source
    with sr.Microphone() as source:
        st.info("Speak now...")
        # Listen for speech and store it in audio_text variable
        audio_text = r.listen(source)
        st.info("Transcribing...")

        try:
            # Use Google Speech Recognition to transcribe speech into text
            text = r.recognize_google(audio_text)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand."
        except sr.RequestError:
            return "Sorry, there was an issue with the speech recognition service."

# Modify the chatbot function to handle both text and speech input from the user
def chatbot(input_text):
    if input_text:
        # Find the most relevant sentence based on text input
        most_relevant_sentence = get_most_relevant_sentence(input_text)
    else:
        # Transcribe speech into text using speech recognition
        input_text = transcribe_speech()
        # Find the most relevant sentence based on transcribed text
        most_relevant_sentence = get_most_relevant_sentence(input_text)
    # Return the answer
    return most_relevant_sentence


# Create a Streamlit app
def main():
    st.title("Chatbot")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")
    # Get the user's input (text or speech)
    input_type = st.radio("Input type:", ("Text", "Speech"))
    if input_type == "Text":
        # Get the user's text input
        input_text = st.text_input("You:")
        # Create a button to submit the text input
        if st.button("Submit"):
            # Call the chatbot function with the text input and display the response
            response = chatbot(input_text)
            st.write("Chatbot: " + response)
    elif input_type == "Speech":
        # Create a button to start speech recognition
        if st.button("Start Recording"):
            # Transcribe speech into text using speech recognition
            input_text = transcribe_speech()
            st.write("Transcription:", input_text)
            # Call the chatbot function with the transcribed text and display the response
            response = chatbot(input_text)
            st.write("Chatbot: " + response)

if __name__ == "__main__":
    main()


