import string
import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import spacy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tag import pos_tag
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import word_tokenize

from config_file import Config
# pylint: disable=no-member
################################################################################
if os.path.exists("eng_config.ini"):
    CONFIG_FILE_NAME = "eng_config.ini"
else:
    CONFIG_FILE_NAME = None
CONFIG = Config(CONFIG_FILE_NAME)
if CONFIG_FILE_NAME is None:
    CONFIG.write_defaults()
CONFIG.get_values_from_config_file()

# Outputs
OUTPUT_HIST = "test_set_hist_"+CONFIG.language[:2]
OUTPUT_PIE = "test_set_pie_"+CONFIG.language[:2]
################################################################################
# function to clear screen after each training sample
def clear():
    os.system('clear')

def load_data(config_obj):
    csvfile = pd.read_csv(config_obj.input_data_file)

    sentences = []
    for sentence in csvfile[config_obj.review_header]:
        sentences.append(sentence)
    return sentences

# cleaning function for english sentences
def clean_words_eng(words, stop_words=()):
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(words)
    words_tagged = []
    for token, tag in pos_tag(words):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        word = lemmatizer.lemmatize(token, pos)
        words_tagged.append(word)
    words_clean = []
    for word in words_tagged:
        word = stemmer.stem(word)
        word = word.lower()
        if word not in stop_words and word not in string.punctuation:
            words_clean.append(word)
    return words_clean

# cleaning function for spanish sentences
def clean_words_esp(words, nlp, stop_words=()):
    stemmer = SnowballStemmer('spanish')

    def lemmatizer(text, pos):
        text = " ".join(text)
        doc = nlp(text, pos)
        result = [word.lemma_ for word in doc]
        return result

    working_dir = os.getcwd()
    os.environ['CLASSPATH'] = working_dir+"/stanford-postagger-full-2020-08-06"
    model = "spanish-ud"
    lang_model = working_dir+"/stanford-postagger-full-2020-08-06/models/"+model+".tagger"
    postagger = StanfordPOSTagger(lang_model, encoding='utf8')

    tokens = []
    pos_list = []
    for token, tag in postagger.tag(words):
        if tag == "NOUN":
            pos = 'n'
        elif tag == "VERB":
            pos = 'v'
        else:
            pos = 'a'
        tokens.append(token)
        pos_list.append(pos)

    words_tagged = lemmatizer(tokens, pos_list)

    words_clean = []
    for word in words_tagged:
        word = stemmer.stem(word)
        word = word.lower()
        if word not in stop_words and word not in string.punctuation:
            words_clean.append(word)
    return words_clean

def clean_split_pad_data(sentences, config_obj):
    """ Clean data, split into train/test groups, and pad sentences for training """
    print("Cleaning data (This may take a few minutes)...")
    if config_obj.language == "english":
        stopwords_list = set(stopwords.words('english'))
        important_words_english = ['above', 'below', 'off', 'over', 'under', 'more',
                                   'most', 'such', 'no', 'nor', 'not', 'only', 'so',
                                   'than', 'too', 'very', 'just', 'but']
        stopwords_list = stopwords_list - set(important_words_english)
        sentences_clean = [" ".join(clean_words_eng(sentence, stopwords_list)) for sentence in sentences]
    elif config_obj.language == "spanish":
        nlp = spacy.load('es_core_news_sm', disable=['ner', 'parser'])
        stopwords_list = set(stopwords.words('spanish'))
        important_words_spanish = ['encima', 'debajo', 'menos', 'abajo', 'más',
                                   'tal', 'no', 'ni', 'solamente', 'entonces',
                                   'que', 'demasiado', 'muy', 'también', 'sólo', 'pero']
        stopwords_list = stopwords_list - set(important_words_spanish)
        sentences_clean = [" ".join(clean_words_esp(sentence, nlp, stopwords_list)) for sentence in sentences]

    # Load tokenizer from json file
    tokenizer_name = "tokenizer_"+config_obj.language[:2]+".json"
    with open(tokenizer_name) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    # Creating padded sequences from train and test data
    sequences_clean = tokenizer.texts_to_sequences(sentences_clean)
    padded = pad_sequences(sequences_clean, maxlen=config_obj.max_len, padding=config_obj.pad_type, truncating=config_obj.trunc_type)

    # Converting the list to numpy array
    sentences_np = np.array(padded)

    return sentences_np

def output_to_class(output_val):
    sentiment = (output_val - 0.5) * 2
    if sentiment == 1:
        output_class = "Positive"
    elif sentiment == -1:
        output_class = "Negative"
    return output_class
################################################################################
print("Welcome to the LSTM Deep Learning Sentiment Classifier. \n")
print("1. Single sample \n2. Batch (csv file) \n")
MODE = input("Single sample or batch (csv file)? ")

if MODE == "1":
    MODEL = load_model(CONFIG.model_name+".h5")
    SAMPLE = [input("Type sample: ")]
    CLEAN_SENTENCES = clean_split_pad_data(SAMPLE, CONFIG)
    CONFIDENCE = round(abs((MODEL.predict(CLEAN_SENTENCES)-0.5) * 200)[0][0], 2)
    Y_PREDICT = (MODEL.predict(CLEAN_SENTENCES)[0][0] > 0.5) * 1
    if Y_PREDICT == 0:
        Y_CLASS = "Negative"
    elif Y_PREDICT == 1:
        Y_CLASS = "Positive"
    print(Y_CLASS, str(CONFIDENCE)+"% confidence")
elif MODE == "2":
    MODEL = load_model(CONFIG.model_name+".h5")
    SENTENCES = load_data(CONFIG)
    CLEAN_SENTENCES = clean_split_pad_data(SENTENCES, CONFIG)
    CONFIDENCE_ARR = np.round(abs((MODEL.predict(CLEAN_SENTENCES)-0.5) * 200), 2)
    Y_PREDICT_ARR = (MODEL.predict(CLEAN_SENTENCES) > 0.5) * 1

    INPUT_DATA_FILE = CONFIG.input_data_file
    CSVFILE = pd.read_csv(INPUT_DATA_FILE)
    CSVFILE.to_csv(INPUT_DATA_FILE.split('.csv')[0]+"_predicted.csv", index=False)

    ADJ_SENTIMENT_LIST = []
    SENTIMENT_DICT = {"Positive": 0, "Negative": 0}
    HUMAN_REVIEW = []
    for idx, (output, confidence) in enumerate(zip(Y_PREDICT_ARR, CONFIDENCE_ARR)):
        if confidence < 50.0:
            output_dict = {
                "sentence": SENTENCES[idx],
                "prediction": output_to_class(output)
                }
            HUMAN_REVIEW.append(output_dict)
        else:
            sentiment = (output - 0.5) * 2
            y_class = output_to_class(output)
            adj_sentiment = sentiment * confidence
            ADJ_SENTIMENT_LIST.append(adj_sentiment[0])
            if sentiment == 1:
                SENTIMENT_DICT["Positive"] += 1
            elif sentiment == -1:
                SENTIMENT_DICT["Negative"] += 1

            CSVFILE = pd.read_csv(INPUT_DATA_FILE.split('.csv')[0]+"_predicted.csv")
            CSVFILE.loc[CSVFILE[CONFIG.review_header] == SENTENCES[idx], "SENTIMENT"] = y_class
            CSVFILE.to_csv(INPUT_DATA_FILE.split('.csv')[0]+"_predicted.csv", index=False)

    # Human Intervention - if predicted has low confidence or doesn't match
    #                      existing label provide a new label
    REVIEW_UNSURE = None
    if len(HUMAN_REVIEW) > 0:
        print("There are {} reviews with questionable tags. Would you like to review those now?".format(len(HUMAN_REVIEW)))
        REVIEW_UNSURE = input("Yes or No (y/n): ")

    if REVIEW_UNSURE == "y":
        for test_review in HUMAN_REVIEW:
            # predict outcome of review
            print("Predicted class confidence is low. Please verify or update class. \n")
            print(test_review["sentence"])
            print("class: ", test_review["prediction"], "\n")
            print("1. Positive \n2. Negative")
            KEYBOARD_INPUT = (int(input("Choose a class: ")) - 2) * -1
            y_class = output_to_class(KEYBOARD_INPUT)

            CSVFILE = pd.read_csv(INPUT_DATA_FILE.split('.csv')[0]+"_predicted.csv")
            CSVFILE.loc[CSVFILE[CONFIG.review_header] == test_review["sentence"], "SENTIMENT"] = y_class
            CSVFILE.to_csv(INPUT_DATA_FILE.split('.csv')[0]+"_predicted.csv", index=False)
            clear()


    print("Plotting results...")
    # Plot sentiment distribution
    plt.hist(ADJ_SENTIMENT_LIST, bins=25, range=(-100, 100))
    plt.title("Overall Sentiment for " + INPUT_DATA_FILE)
    plt.xlabel("Confidence x Sentiment")
    plt.ylabel("Counts")
    plt.savefig(OUTPUT_HIST + ".png", dpi=1250)
    plt.cla()

    plt.pie(SENTIMENT_DICT.values(), labels=SENTIMENT_DICT.keys(), autopct='%1.1f%%',
            startangle=90)
    plt.axis('equal')
    plt.title("Response Sentiments for " + INPUT_DATA_FILE)
    plt.savefig(OUTPUT_PIE + ".png", dpi=1250)
